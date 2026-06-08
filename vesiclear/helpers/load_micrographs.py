from vesicle_picker import external_import

def load_micrographs(cryosparc_parameters):
    cs = external_import.load_cryosparc(cryosparc_parameters["csparc_input_login"])
    project = cs.find_project(cryosparc_parameters["csparc_input_PID"])
    micrographs_dataset = external_import.micrographs_from_csparc(
        cs=cs,
        project_id=cryosparc_parameters["csparc_input_PID"],
        job_id=cryosparc_parameters["csparc_input_JID"],
        job_type=cryosparc_parameters["csparc_input_type"]
    )
    micrographs = micrographs_dataset.to_records()
    return project, micrographs
