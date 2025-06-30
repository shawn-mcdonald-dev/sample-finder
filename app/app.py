from scraper.freesound_api import FreesoundClient

client = FreesoundClient()
client.fetch_and_store(query="ambient pad", max_results=10)
