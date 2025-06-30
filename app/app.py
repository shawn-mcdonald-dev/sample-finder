from scraper.freesound_api import FreesoundClient

client = FreesoundClient()
client.fetch_and_store(query="jazz", max_results=10)
