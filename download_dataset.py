from icrawler.builtin import BingImageCrawler

def download_images(keywords, folder, num_images_per_keyword=100):

    crawler = BingImageCrawler(
        storage={'root_dir': folder}
    )

    for keyword in keywords:

        print(f"\nDownloading: {keyword}")

        crawler.crawl(
            keyword=keyword,
            max_num=num_images_per_keyword
        )

# LANDMARK KEYWORDS

landmarks = {

    # TAJ MAHAL

    "taj_mahal": [

        "Taj Mahal India",
        "Taj Mahal sunrise",
        "Taj Mahal sunset",
        "Taj Mahal night view",
        "Taj Mahal front view",
        "Taj Mahal aerial view",
        "Taj Mahal architecture",
        "Taj Mahal reflection",
        "Taj Mahal tourists",
        "Taj Mahal close up",
        "Taj Mahal garden view",
        "Taj Mahal full image",
        "Taj Mahal side angle",
        "Taj Mahal drone shot",
        "Taj Mahal clear sky"

    ],

    # EIFFEL TOWER

    "eiffel_tower": [

        "Eiffel Tower Paris",
        "Eiffel Tower night lights",
        "Eiffel Tower aerial view",
        "Eiffel Tower sunset",
        "Eiffel Tower tourists",
        "Eiffel Tower architecture",
        "Eiffel Tower close up",
        "Eiffel Tower drone shot",
        "Eiffel Tower city view",
        "Eiffel Tower top view",
        "Eiffel Tower full image",
        "Eiffel Tower side angle",
        "Eiffel Tower blue sky",
        "Eiffel Tower evening",
        "Eiffel Tower river view"

    ],

    # QUTUB MINAR

    "qutub_minar": [

        "Qutub Minar Delhi",
        "Qutub Minar full view",
        "Qutub Minar close up",
        "Qutub Minar architecture",
        "Qutub Minar tourists",
        "Qutub Minar blue sky",
        "Qutub complex",
        "Qutub Minar side angle",
        "Qutub Minar aerial",
        "Qutub Minar drone shot",
        "Qutub Minar historical monument",
        "Qutub Minar garden",
        "Qutub Minar daytime",
        "Qutub Minar sunset",
        "Qutub Minar stone texture"

    ],

    # GREAT WALL

    "great_wall": [

        "Great Wall of China",
        "Great Wall mountains",
        "Great Wall aerial",
        "Great Wall sunset",
        "Great Wall drone shot",
        "Great Wall tourists",
        "Great Wall architecture",
        "Great Wall close up",
        "Great Wall winter",
        "Great Wall green mountains",
        "Great Wall full image",
        "Great Wall cloudy sky",
        "Great Wall sunrise",
        "Great Wall pathway",
        "Great Wall panoramic"

    ],

    # HAWA MAHAL

    "hawa_mahal": [

        "Hawa Mahal Jaipur",
        "Hawa Mahal front view",
        "Hawa Mahal architecture",
        "Hawa Mahal windows",
        "Hawa Mahal tourists",
        "Hawa Mahal evening",
        "Hawa Mahal side angle",
        "Hawa Mahal close up",
        "Hawa Mahal aerial",
        "Hawa Mahal street view",
        "Hawa Mahal daytime",
        "Hawa Mahal pink city",
        "Hawa Mahal clear sky",
        "Hawa Mahal historical monument",
        "Hawa Mahal full image"

    ],

    # GOLDEN TEMPLE

    "golden_temple": [

        "Golden Temple Amritsar",
        "Golden Temple reflection",
        "Golden Temple night lights",
        "Golden Temple sunrise",
        "Golden Temple architecture",
        "Golden Temple aerial view",
        "Golden Temple tourists",
        "Golden Temple lake view",
        "Harmandir Sahib full image",
        "Golden Temple close up",
        "Golden Temple prayer",
        "Golden Temple daytime",
        "Golden Temple evening",
        "Golden Temple panoramic",
        "Golden Temple blue sky"

    ],

    # STATUE OF LIBERTY

    "statue_of_liberty": [

        "Statue of Liberty New York",
        "Statue of Liberty skyline",
        "Statue of Liberty sunset",
        "Statue of Liberty aerial",
        "Statue of Liberty close up",
        "Statue of Liberty tourists",
        "Statue of Liberty river view",
        "Statue of Liberty drone shot",
        "Statue of Liberty daytime",
        "Statue of Liberty evening",
        "Statue of Liberty architecture",
        "Statue of Liberty full image",
        "Statue of Liberty blue sky",
        "Statue of Liberty side angle",
        "Statue of Liberty panoramic"

    ]
}

# DOWNLOAD TRAIN & VALIDATION DATA

for name, keywords in landmarks.items():

    # TRAIN DATA
    download_images(
        keywords,
        f"dataset/train/{name}",
        100
    )

    # VALIDATION DATA
    download_images(
        keywords,
        f"dataset/val/{name}",
        20
    )

print("\nDataset Download Completed Successfully!")