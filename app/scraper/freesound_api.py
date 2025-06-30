import os
import time
import logging
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FreesoundClient:
    BASE_URL = "https://freesound.org/apiv2"
    SEARCH_ENDPOINT = "/search/text/"
    SOUND_ENDPOINT = "/sounds/{sound_id}/"

    def __init__(self, api_key: Optional[str] = None, download_dir: str = "data/raw/audio", metadata_path: str = "data/raw/metadata/freesound_metadata.jsonl"):
        self.api_key = api_key or os.getenv("FREESOUND_API_KEY")
        if not self.api_key:
            raise ValueError("FREESOUND_API_KEY not found in environment or passed explicitly.")

        self.session = requests.Session()
        self.session.params = {"token": self.api_key}

        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    def search_samples(self, query: str, max_results: int = 20, filter_tags: Optional[str] = None) -> List[Dict]:
        """Search Freesound for audio samples by text query."""
        logger.info(f"Searching Freesound for query='{query}' with max_results={max_results}")

        results = []
        url = self.BASE_URL + self.SEARCH_ENDPOINT
        params = {
            "query": query,
            "fields": "id,name,previews,download,username,tags,duration,type,samplerate,bitrate,bpm,key,license",
            "page_size": 20,
        }
        if filter_tags:
            params["filter"] = f"tag:{filter_tags}"

        count = 0
        next_url = url
        while next_url and count < max_results:
            response = self.session.get(next_url, params=params)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch results: {response.status_code}")
                break

            data = response.json()
            page_results = data.get("results", [])
            results.extend(page_results)
            count += len(page_results)

            next_url = data.get("next")
            params = {}  # params only go with the first request
            time.sleep(1)  # rate limiting

        logger.info(f"Found {len(results)} results.")
        return results[:max_results]

    def download_sample(self, sample: Dict, overwrite: bool = False) -> Optional[str]:
        """Download a single sample to the local filesystem."""
        preview_url = sample.get("previews", {}).get("preview-hq-mp3") or sample.get("previews", {}).get("preview-hq-ogg")
        if not preview_url:
            logger.warning(f"No preview URL found for sample: {sample.get('name')}")
            return None

        sample_id = sample["id"]
        filename = f"{sample_id}_{sample['name'].replace(' ', '_')}.mp3"
        filepath = self.download_dir / filename

        if filepath.exists() and not overwrite:
            logger.info(f"File already exists, skipping: {filename}")
            return str(filepath)

        try:
            logger.info(f"Downloading sample: {sample['name']} -> {filepath.name}")
            with self.session.get(preview_url, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None

    def save_metadata(self, samples: List[Dict]):
        """Append metadata to JSONL file."""
        logger.info(f"Saving metadata for {len(samples)} samples.")
        with open(self.metadata_path, "a", encoding="utf-8") as f:
            for sample in samples:
                json.dump(sample, f)
                f.write("\n")

    def fetch_and_store(self, query: str, max_results: int = 20):
        """Search, download and save metadata for a batch of samples."""
        samples = self.search_samples(query=query, max_results=max_results)
        for sample in samples:
            self.download_sample(sample)
        self.save_metadata(samples)
