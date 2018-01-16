
from util.data_processing import *

if True:
    print("Loading data SNLI")
    training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"], snli=True)
    dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
    test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)

    print("Loading data MNLI")
    training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
    dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
    dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])

    test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"], shuffle = False)
    test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"], shuffle = False)
    #shared_content = load_mnli_shared_content()

    print("Loading embeddings")
    #indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([dev_snli, dev_matched])
    #indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([training_mnli, training_snli, dev_matched, dev_mismatched, test_matched, test_mismatched, dev_snli, test_snli])

    # print('start saving wn rel')
    # #save_wordnet_rel([dev_snli, dev_matched], indices_to_words)
    # save_wordnet_rel([training_mnli, training_snli, dev_matched, dev_mismatched, test_matched, test_mismatched, dev_snli, test_snli])
    # print('finish saving wn rel')

    print('start fixing wn rel')
    #fix_wordnet_rel([dev_snli, dev_matched])
    fix_wordnet_rel([training_mnli, training_snli, dev_matched, dev_mismatched, test_matched, test_mismatched, dev_snli, test_snli])
