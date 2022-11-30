from news_traveler_document_similarity.tfidf_similarity import process_tfidf_similarity


def test_process_tfidf_similarity():
    base_document = "This summer, the 235-foot research vessel Marcus G. Langseth set out into the ocean off the Pacific Northwest. Trailing the ship were four electronic serpents, each five miles in length. These cables were adorned with scientific instruments able to peer into the beating heart of a monster a mile below the waves: Axial Seamount, a volcanic mountain."
    document_to_compare = "SKAFTAFELL, Iceland — Just north of here, on the far side of the impenetrable Vatnajokull ice sheet, lava is spewing from a crack in the earth on the flanks of Bardarbunga, one of Iceland’s largest volcanoes. By volcanologists’ standards, it is a peaceful eruption, the lava merely spreading across the landscape as gases bubble out of it. For now, those gases — especially sulfur dioxide, which can cause respiratory and other problems — are the main concern, prompting health advisories in the capital, Reykjavik, 150 miles to the west, and elsewhere around the country."

    result = process_tfidf_similarity(base_document, document_to_compare)

    assert result >= 0.0 and result <= 1.0
