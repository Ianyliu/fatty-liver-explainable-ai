from sampling_marginal_relation_pipeline import LIME_all_subj_pipeline
testing_09 = LIME_all_subj_pipeline(
    test_data_id = '09', 
    cuda_device_no = 1
)
testing_09.get_marginal_relations_of_all_subj()
