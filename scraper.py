import requests
from bs4 import BeautifulSoup
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
import time
import logging
import re
import gspread.utils # Ensure this is imported

# --- Authentication ---
# Authenticate using the Service Account JSON file
SERVICE_ACCOUNT_FILE = "service_account_creds.json" # The file we downloaded

print("Authenticating with Google Service Account...")
try:
    # gspread.service_account() is the new way to auth
    gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
    print("Authentication successful!")
except Exception as e:
    print(f"!!! FAILED to authenticate using {SERVICE_ACCOUNT_FILE} !!!")
    print("Make sure the file is in the same directory and you shared your Sheet with the client_email.")
    print(f"Error: {e}")
    gc = None # Set gc to None so the script stops

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Colab - %(levelname)s - %(message)s')

print("Setup Complete. Proceed to the Configuration cell.")

# @title Configuration (Edit these values)

# Dictionary: { "Team Name for Tab": ["URL1", "URL2", ...] }
TEAM_URLS = {
    "SISSA 1 - NOSBO": ["https://nosbo.netstand.nl/teams/view/608"],
    "SISSA 2 - NOSBO": ["https://nosbo.netstand.nl/teams/view/624"],
    "SISSA 3 - NOSBO": ["https://nosbo.netstand.nl/teams/view/617"],
    "SISSA 4 - NOSBO": ["https://nosbo.netstand.nl/teams/view/616"],
    "SISSA 5 - NOSBO": ["https://nosbo.netstand.nl/teams/view/639"],
    "SISSA 6 - NOSBO": ["https://nosbo.netstand.nl/teams/view/631"],

    "SISSA - Open Beker": ["https://nosbo.netstand.nl/teams/view/652"],
    "SISSA - 1900 Beker": ["https://nosbo.netstand.nl/teams/view/665"],
    "SISSA - 1700 Beker": ["https://nosbo.netstand.nl/teams/view/668"],

    "SISSA 1 - KNSB": ["https://knsb.netstand.nl/teams/view/6259"],
    "SISSA 2 - KNSB": ["https://knsb.netstand.nl/teams/view/6251"],
    "SISSA 3 - KNSB": ["https://knsb.netstand.nl/teams/view/6452"],
    "SISSA 4 - KNSB": ["https://knsb.netstand.nl/teams/view/6459"],

    "SISSA - KNSB Beker": ["https://knsb.netstand.nl/teams/view/6610"]
}

# Columns to extract directly from the table (used during initial processing)
DESIRED_COLUMNS = ['Speler', 'Rating', 'Pt.', 'TPR', 'W-We', 'Kl.']

# Google Sheet Configuration
# !!! IMPORTANT: The user running this notebook MUST have EDIT access to this Google Sheet !!!
GOOGLE_SHEET_NAME = "Oosthoek beker" # <--- User should ensure this is correct
SUMMARY_FULL_TAB_NAME = "Summary - Full"
SUMMARY_OOSTHOEK_TAB_NAME = "Summary - Oosthoek"
MIN_GAMES_FOR_OOSTHOEK = 5

# Google Sheet Formatting
AUTO_RESIZE_COLUMNS = True
CHAR_WIDTH_MULTIPLIER = 7
PADDING = 25
MIN_WIDTH = 40
MAX_WIDTH = 400

# Summary Tab Column Name Customization
SUMMARY_COLUMN_NAMES = {
    'Speler': 'Speler',
    'Total_Points': 'Punten',
    'Total_Games': 'Partijen',
    'Win_Percentage': 'Win %'
}

print("Configuration loaded.")

# @title Helper Functions (Scraping, Processing, Sheets)

# --- Helper Functions ---

def find_results_table(soup):
    """Finds the table immediately following the 'Persoonlijke Resultaten' header."""
    header = soup.find(lambda tag: tag.name in ['h2', 'h3', 'h4'] and 'Persoonlijke Resultaten' in tag.get_text())
    if not header:
        # Try alternative header text common on KNSB site
        header = soup.find(lambda tag: tag.name in ['h2', 'h3', 'h4'] and 'Persoonlijke scores' in tag.get_text())
        if not header:
            logging.warning("Could not find 'Persoonlijke Resultaten' or 'Persoonlijke scores' header.")
            return None

    table = header.find_next_sibling('table')
    if not table:
        parent = header.parent
        siblings = header.find_next_siblings()
        for sibling in siblings:
            if sibling.name == 'table':
                table = sibling
                break
            if sibling.find('table'):
                 table = sibling.find('table')
                 break
    if not table:
         logging.warning("Could not find table immediately following the header.")
    return table

def parse_table(table_soup):
    """Parses an HTML table soup into a pandas DataFrame, handling potential row headers (th) in tbody."""
    headers = []
    header_row = table_soup.find('thead').find('tr') if table_soup.find('thead') else table_soup.find('tr')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
    if not headers:
        logging.warning("Could not find table headers.")
        return pd.DataFrame()

    rows = []
    tbody = table_soup.find('tbody')
    if not tbody:
        tbody = table_soup
        logging.warning("Could not find table body (tbody), trying table directly.")
    if not tbody:
        logging.warning("Could not find table body (tbody) or rows directly under table.")
        return pd.DataFrame(columns=headers)

    for tr in tbody.find_all('tr'):
        if 'sumrow' in tr.get('class', []) or tr.find('td', {'colspan': True}) or tr.find('th', {'colspan': True}):
            logging.info(f"Skipping potential summary/footer row: {tr.get_text(strip=True, separator='|')}")
            continue
        cells = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
        if len(cells) == len(headers):
            rows.append(cells)
        else:
            logging.warning(f"Skipping row. Cell count ({len(cells)}) != Header count ({len(headers)}). Row data: {cells}. Headers: {headers}")

    if not rows:
        logging.warning("No valid data rows found after filtering.")
        return pd.DataFrame(columns=headers)

    df = pd.DataFrame(rows, columns=headers)

    # Rename points column if needed (e.g., 'Prt' or 'Pt' -> 'Pt.')
    points_col_map = {'Prt': 'Pt.', 'Pt': 'Pt.'}
    df = df.rename(columns=lambda c: points_col_map.get(c, c))

    if headers and headers[0] == '':
        logging.info("Renaming blank first column header to 'Board'.")
        df = df.rename(columns={'': 'Board'})

    if 'Speler' in df.columns:
       df = df[df['Speler'].astype(str).str.strip() != '']
    return df


def process_team_data(df, desired_columns):
    """Processes the raw DataFrame: selects columns, calculates games."""
    if df.empty:
        return df, []

    # Identify round columns (numeric headers, or R+number)
    round_columns = [col for col in df.columns if re.fullmatch(r'\d+', col) or re.fullmatch(r'R\d+', col, re.IGNORECASE)]
    logging.info(f"Identified round columns: {round_columns}")

    # --- Find and Rename 'Speler' column ---
    speler_col = next((col for col in df.columns if col.lower() == 'speler'), None)
    if not speler_col:
        logging.error("'Speler' column (or similar) not found in table, cannot proceed with this table.")
        return pd.DataFrame(), []
    if speler_col != 'Speler':
        logging.info(f"Using column '{speler_col}' as 'Speler'.")
        df = df.rename(columns={speler_col: 'Speler'})

    # --- Select Desired Columns ---
    available_desired_columns = [col for col in desired_columns if col in df.columns]
    missing_desired = [col for col in desired_columns if col not in df.columns]
    if missing_desired:
        logging.warning(f"Desired columns not found in the table: {missing_desired}")
    # Ensure 'Speler' is always included, even if not explicitly desired (needed for grouping)
    if 'Speler' not in available_desired_columns:
        available_desired_columns.insert(0,'Speler')
    # Ensure Pt. and Games columns are present if they exist, as they are needed downstream
    if 'Pt.' in df.columns and 'Pt.' not in available_desired_columns:
         available_desired_columns.append('Pt.')
    # Note: 'Games' is calculated, not selected initially

    processed_df = df[list(dict.fromkeys(available_desired_columns))].copy() # Select unique columns

    # --- Calculate 'Games' ---
    def count_games(row):
        """Counts played games based on round column entries."""
        count = 0
        # Iterate through the identified round columns for this specific row
        for col in round_columns:
            # Use .get() on the row (which is a Series in apply) for safety
            value = str(row.get(col, '')).strip()

            # Define conditions for a game that was *NOT* played or shouldn't be counted
            is_empty = value == ''
            is_placeholder = value == '-'
            is_bye = 'bye' in value.lower() or 'vrij' in value.lower() # Dutch 'vrij'
            is_no_opponent = 'z.t.' in value.lower() # Dutch 'zonder tegenstander'
            is_not_played = 'n.g.' in value.lower() # Dutch 'niet gespeeld'
            is_reglementair_no_show = value == '0-0' # Forfeit by both / no show

            # If NONE of the non-played conditions are met, count it as a played game.
            # This logic includes '0', '1', '½', '1-0', '0-1', '½-½' etc.
            if not (is_empty or is_placeholder or is_bye or is_no_opponent or is_not_played or is_reglementair_no_show):
                 count += 1
            # Optional: For debugging, log which values *were not* counted
            # else:
            #    if value: # Log only if it wasn't empty
            #       logging.debug(f"Player {row.get('Speler', 'Unknown')}, Col {col}: Value '{value}' NOT counted as game.")

        return count

    if round_columns:
        # Ensure all columns used in apply exist in the df to avoid errors
        apply_cols = [col for col in round_columns if col in df.columns]
        if apply_cols:
             processed_df['Games'] = df.apply(lambda row: count_games(row), axis=1) # Pass the whole row
        else:
             processed_df['Games'] = 0
             logging.warning("No applicable round columns found in DataFrame rows, setting 'Games' to 0.")
    else:
        processed_df['Games'] = 0 # No round columns means 0 games
        logging.warning("No round columns identified from headers, setting 'Games' to 0.")

    # --- Data Cleaning ---
    processed_df['Speler'] = processed_df['Speler'].astype(str).str.strip()

    # Clean 'Pt.' column (ensure it exists)
    if 'Pt.' in processed_df.columns:
        processed_df['Pt.'] = processed_df['Pt.'].astype(str).str.replace(',', '.', regex=False)
        processed_df['Pt.'] = processed_df['Pt.'].replace({'½': 0.5})
        processed_df['Pt.'] = pd.to_numeric(processed_df['Pt.'], errors='coerce')
        processed_df['Pt.'] = processed_df['Pt.'].fillna(0.0) # Replace errors/NaN with 0 points
    else:
        # If 'Pt.' wasn't found initially, create it with 0 to avoid errors later
        logging.warning("'Pt.' column not found. Creating it with 0 points.")
        processed_df['Pt.'] = 0.0

    # Ensure 'Games' is integer
    processed_df['Games'] = processed_df['Games'].astype(int)

    # Return only the columns we definitely processed or need downstream ('Speler', 'Pt.', 'Games' + others if kept)
    final_cols = ['Speler', 'Pt.', 'Games']
    for col in available_desired_columns:
         if col not in final_cols and col in processed_df.columns:
              final_cols.append(col)

    return processed_df[final_cols], round_columns

def auto_resize_columns(worksheet, df):
    """Resizes columns based on content length."""
    if df.empty:
        logging.warning(f"Skipping auto-resize for empty DataFrame in sheet '{worksheet.title}'.")
        return

    requests = []
    spreadsheet = worksheet.spreadsheet

    for i, col_name in enumerate(df.columns):
        try:
            col_values = df[col_name].astype(str).tolist()
            all_lengths = [len(s) for s in col_values]
            all_lengths.append(len(str(col_name)))
            max_len = max(all_lengths) if all_lengths else 1
            pixel_width = max(MIN_WIDTH, min(MAX_WIDTH, int(max_len * CHAR_WIDTH_MULTIPLIER + PADDING)))
            requests.append({
                "updateDimensionProperties": {
                    "range": {"sheetId": worksheet.id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
                    "properties": {"pixelSize": pixel_width},
                    "fields": "pixelSize"
                }
            })
        except Exception as e:
            logging.error(f"Error calculating width for column '{col_name}' in sheet '{worksheet.title}': {e}")

    if requests:
        try:
            logging.info(f"Applying auto-resize to {len(requests)} columns in sheet '{worksheet.title}'...")
            spreadsheet.batch_update({"requests": requests})
        except Exception as e:
            logging.error(f"Failed to batch update column widths for sheet '{worksheet.title}': {e}")


def write_to_gsheet(client, sheet_name, tab_name, df):
    """Writes DataFrame to Google Sheet tab, handling errors."""
    try:
        # Use the authenticated client 'gc' passed from the setup cell
        spreadsheet = client.open(sheet_name)
        try:
            worksheet = spreadsheet.worksheet(tab_name)
            logging.info(f"Updating existing tab: '{tab_name}'")
            worksheet.clear()
            worksheet.format('A1:Z', {'backgroundColor': {'red': 1, 'green': 1, 'blue': 1}, 'textFormat': {'bold': False}})
        except gspread.WorksheetNotFound:
            logging.info(f"Creating new tab: '{tab_name}'")
            max_rows = max(len(df) + 1, 2)
            max_cols = max(len(df.columns), 1)
            worksheet = spreadsheet.add_worksheet(title=tab_name, rows=max_rows, cols=max_cols)

        df.columns = df.columns.astype(str)
        set_with_dataframe(worksheet, df, include_index=False, include_column_header=True, resize=False)
        logging.info(f"Successfully wrote data to tab '{tab_name}'.")

        if AUTO_RESIZE_COLUMNS:
            auto_resize_columns(worksheet, df)

        try:
            worksheet.freeze(rows=1)
        except Exception as e:
             logging.warning(f"Could not freeze header row for sheet '{tab_name}': {e}")

    except gspread.exceptions.SpreadsheetNotFound:
         logging.error(f"Google Sheet '{sheet_name}' not found. Make sure the name is correct AND the user running this notebook has edit access.")
    except gspread.exceptions.APIError as e:
         logging.error(f"Google API Error writing to tab '{tab_name}'. Check permissions or quotas: {e}")
    except Exception as e:
        logging.error(f"Unexpected error writing to Google Sheet tab '{tab_name}': {e}")

print("Helper functions defined.")

# @title Run Scraper and Generate Sheets

# Use the authenticated client 'gc' from the setup cell
gsheet_client = gc

processed_teams_data = {}
logging.info("Starting chess data scraping process...")

# --- Scraping and Per-Team Processing/Combining ---
for team_name, url_list in TEAM_URLS.items():
    logging.info(f"--- Processing Team: {team_name} ---")
    team_phase_dfs = []
    for url in url_list:
        logging.info(f"Fetching URL: {url}")
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch URL {url} for team {team_name}: {e}")
            time.sleep(2); continue
        soup = BeautifulSoup(response.content, 'html.parser')
        table = find_results_table(soup)
        if table:
            raw_df = parse_table(table)
            if not raw_df.empty:
                processed_df, _ = process_team_data(raw_df.copy(), DESIRED_COLUMNS)
                if not processed_df.empty:
                    team_phase_dfs.append(processed_df)
                else: logging.warning(f"No processable player data found for URL: {url}")
            else: logging.warning(f"Parsed table was empty for URL: {url}")
        else: logging.warning(f"Could not find results table for URL: {url}")
        time.sleep(1)

    # Combine data for the current team
    if not team_phase_dfs:
        logging.warning(f"No data collected for team {team_name}. Skipping.")
        continue
    combined_team_df = pd.concat(team_phase_dfs, ignore_index=True)
    logging.info(f"Aggregating results for {team_name} across {len(team_phase_dfs)} phases/URLs...")
    agg_logic = {'Pt.': 'sum', 'Games': 'sum'}
    other_cols_to_keep = [col for col in DESIRED_COLUMNS if col not in ['Speler', 'Pt.', 'Games'] and col in combined_team_df.columns]
    for col in other_cols_to_keep: agg_logic[col] = 'first'

    # Ensure essential columns exist before aggregation if they were missing from all phases
    if 'Pt.' not in combined_team_df.columns: combined_team_df['Pt.'] = 0.0
    if 'Games' not in combined_team_df.columns: combined_team_df['Games'] = 0

    # Handle potential empty Speler groups if data is inconsistent
    if 'Speler' in combined_team_df.columns:
        final_team_df = combined_team_df.dropna(subset=['Speler']).groupby('Speler', as_index=False).agg(agg_logic)

        final_cols_order = ['Speler'] + other_cols_to_keep + ['Pt.', 'Games']
        final_cols_order = [col for col in final_cols_order if col in final_team_df.columns]
        final_team_df = final_team_df[final_cols_order]
        processed_teams_data[team_name] = final_team_df
        logging.info(f"Finished processing team {team_name}. Final player count: {len(final_team_df)}")
    else:
         logging.error(f"Cannot aggregate team {team_name} as 'Speler' column is missing.")


# --- Google Sheets Output ---
if not processed_teams_data:
    logging.warning("No data successfully processed for any teams. Exiting.")
elif not gsheet_client:
     logging.error("Google Sheets client not authenticated. Cannot write data.")
else:
    logging.info("Writing data to Google Sheets...")
    # Write per-team tabs
    for team_name, final_team_df in processed_teams_data.items():
        if not final_team_df.empty:
            if 'Pt.' in final_team_df.columns:
                 final_team_df = final_team_df.sort_values(by='Pt.', ascending=False)
            write_to_gsheet(gsheet_client, GOOGLE_SHEET_NAME, team_name, final_team_df)
            time.sleep(3)
        else: logging.warning(f"Skipping sheet write for team '{team_name}' (empty).")

    # Overall Aggregation
    logging.info("Aggregating data across all teams for final summary...")
    all_teams_list = [df for df in processed_teams_data.values() if not df.empty]
    if not all_teams_list:
        logging.warning("No non-empty team dataframes to aggregate for summary.")
    else:
        all_teams_combined_df = pd.concat(all_teams_list, ignore_index=True)
        required_agg_cols = ['Speler', 'Pt.', 'Games']
        if not all(col in all_teams_combined_df.columns for col in required_agg_cols):
             logging.error(f"Missing required columns for final summary: {required_agg_cols}. Aborting.")
        else:
            # Group by player ACROSS ALL TEAMS
            summary = all_teams_combined_df.groupby('Speler').agg(
                Total_Points=('Pt.', 'sum'),
                Total_Games=('Games', 'sum')
            ).reset_index()
            summary['Win_Percentage'] = summary.apply(lambda r: (r['Total_Points']/r['Total_Games']*100) if r['Total_Games'] > 0 else 0.0, axis=1).round(2)
            summary = summary.sort_values(by=['Win_Percentage', 'Total_Games'], ascending=[False, False])

            # Filter for Oosthoek
            summary_oosthoek = summary[summary['Total_Games'] >= MIN_GAMES_FOR_OOSTHOEK].copy()
            logging.info(f"Filtered summary for '{SUMMARY_OOSTHOEK_TAB_NAME}' has {len(summary_oosthoek)} players.")

            # Rename Columns
            summary_full_output_df = summary.copy()
            summary_oosthoek_output_df = summary_oosthoek.copy()
            if SUMMARY_COLUMN_NAMES:
                rename_map = {k: v for k, v in SUMMARY_COLUMN_NAMES.items() if k in summary.columns}
                if rename_map:
                     logging.info(f"Renaming summary columns: {rename_map}")
                     summary_full_output_df = summary_full_output_df.rename(columns=rename_map)
                     summary_oosthoek_output_df = summary_oosthoek_output_df.rename(columns=rename_map)

            # Write Summary Tabs
            logging.info(f"Writing full summary to tab '{SUMMARY_FULL_TAB_NAME}'...")
            write_to_gsheet(gsheet_client, GOOGLE_SHEET_NAME, SUMMARY_FULL_TAB_NAME, summary_full_output_df)
            time.sleep(3)
            logging.info(f"Writing filtered summary to tab '{SUMMARY_OOSTHOEK_TAB_NAME}'...")
            write_to_gsheet(gsheet_client, GOOGLE_SHEET_NAME, SUMMARY_OOSTHOEK_TAB_NAME, summary_oosthoek_output_df)

            logging.info("Script finished successfully.")

    # --- Final Sheet Cleanup and Reordering ---
    if gsheet_client and processed_teams_data: # Only run if client is valid and some data was processed
        try:
            logging.info("Performing final sheet cleanup and reordering...")
            spreadsheet = gsheet_client.open(GOOGLE_SHEET_NAME)

            # 1. Delete empty "Sheet1" if it exists
            try:
                default_sheet = spreadsheet.worksheet("Sheet1")
                # Check if it's effectively empty (no values or only empty strings/rows)
                all_vals = default_sheet.get_all_values()
                is_empty = not all_vals or all(not any(cell for cell in row) for row in all_vals)

                if is_empty:
                    logging.info("Found empty 'Sheet1'. Deleting it...")
                    spreadsheet.del_worksheet(default_sheet)
                    time.sleep(1) # Give API a moment
                else:
                    logging.info("'Sheet1' exists but contains data. Skipping deletion.")
            except gspread.WorksheetNotFound:
                logging.info("'Sheet1' not found. No need to delete.")
            except Exception as e:
                logging.warning(f"Could not check or delete 'Sheet1': {e}")

            # 2. Reorder sheets: Summaries first, then others
            desired_order_names = [SUMMARY_OOSTHOEK_TAB_NAME, SUMMARY_FULL_TAB_NAME]
            desired_order_worksheets = []
            other_worksheets = []

            all_current_worksheets = spreadsheet.worksheets() # Get fresh list after potential deletion

            # Separate desired sheets and others
            for ws in all_current_worksheets:
                if ws.title in desired_order_names:
                    # Store in correct order based on desired_order_names
                    if ws.title == SUMMARY_FULL_TAB_NAME:
                        desired_order_worksheets.insert(0, ws) # Ensure Full is first
                    elif ws.title == SUMMARY_OOSTHOEK_TAB_NAME:
                        if len(desired_order_worksheets) > 0 and desired_order_worksheets[0].title == SUMMARY_FULL_TAB_NAME:
                             desired_order_worksheets.insert(1, ws) # Ensure Oosthoek is second
                        else: # If Full wasn't found put Oosthoek first
                             desired_order_worksheets.insert(0, ws)
                else:
                    other_worksheets.append(ws)

            # Combine the lists
            final_order_list = desired_order_worksheets + other_worksheets

            # Only reorder if the list isn't trivial and order changed
            current_order_titles = [ws.title for ws in all_current_worksheets]
            final_order_titles = [ws.title for ws in final_order_list]

            if len(final_order_list) > 1 and current_order_titles != final_order_titles:
                logging.info(f"Reordering sheets to: {final_order_titles}")
                spreadsheet.reorder_worksheets(final_order_list)
            else:
                logging.info("Sheet order is already correct or only one sheet exists. No reordering needed.")

        except gspread.exceptions.APIError as e:
            logging.error(f"Google API Error during sheet cleanup/reordering: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during sheet cleanup/reordering: {e}")


print("--- Execution Finished ---")
