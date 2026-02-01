# Oosthoek Beker Scraper

This project scrapes chess competition data (KNSB & NOSBO) for SISSA teams and updates a Google Sheet with the results. It also generates a specific summary for website integration.

## Setup

1.  **Dependencies**: Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    *   Create a `.env` file in the root directory (use `.env.template` as a guide).
    *   Add your Google Service Account JSON string to the `GSPREAD_CREDS_JSON` variable.
    *   *Note: The `.env` file is gitignored for security.*

## Running the Scraper

Run the script using Python:
```bash
python scraper.py
```

This will:
1.  Scrape the latest results from the configured URLs (Netstand).
2.  Process and aggregate the data.
3.  Update the Google Sheet tabs (Teams, Summaries).
4.  Update the **`Website-Export`** tab (Players with >= 50% score).

## Website Integration (WordPress)

To display the standings on your WordPress site automatically:

1.  **Publish to Web**:
    *   In Google Sheets: `File > Share > Publish to web`.
    *   Select the **`Website-Export`** tab.
    *   Choose **Comma-separated values (.csv)** as the format.
    *   Click Publish and copy the generated link.

2.  **Add to WordPress**:
    *   Create a **Custom HTML** block on your page.
    *   Paste the code below (replace `YOUR_CSV_LINK_HERE` with your link):

```html
<div id="standings-container">
    <p>Loading standings...</p>
</div>

<style>
    #standings-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px; }
    #standings-table th, #standings-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    #standings-table th { background-color: #f2f2f2; font-weight: bold; }
    #standings-table tr:nth-child(even) { background-color: #fafafa; }
</style>

<script>
    const csvUrl = 'YOUR_CSV_LINK_HERE'; // <--- PASTE LINK HERE

    fetch(csvUrl)
        .then(response => response.text())
        .then(csvText => {
            const rows = csvText.split('\n').map(row => row.split(','));
            let html = '<table id="standings-table"><thead><tr>';
            
            // Header
            rows[0].forEach(header => {
                html += `<th>${header.replace(/"/g, '')}</th>`;
            });
            html += '</tr></thead><tbody>';

            // Rows
            for (let i = 1; i < rows.length; i++) {
                if (rows[i].join('').trim() === '') continue;
                html += '<tr>';
                rows[i].forEach(cell => {
                    html += `<td>${cell.replace(/"/g, '')}</td>`;
                });
                html += '</tr>';
            }
            html += '</tbody></table>';
            document.getElementById('standings-container').innerHTML = html;
        })
        .catch(error => {
            document.getElementById('standings-container').innerHTML = '<p>Error loading data.</p>';
            console.error('Error:', error);
        });
</script>
```
