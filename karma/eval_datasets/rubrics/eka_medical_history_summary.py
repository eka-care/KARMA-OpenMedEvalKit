from datetime import datetime
import json
import os
from typing import Dict, Any, List

import yaml
from pydantic import BaseModel, Field

from karma.data_models.dataloader_iterable import (
    DataLoaderIterable,
    ConversationTurn,
    Conversation,
    RubricCriteria,
)
from karma.eval_datasets.rubrics.rubric_base_dataset import RubricBaseDataset
from karma.registries.dataset_registry import register_dataset

# base_default_system_prompt = """You are a helpful assistant. Generate a succinct summary of the medical history for the last 6 months."""
base_default_system_prompt = """You are a medical copilot with expansive medical knowledge that should mimic an experienced medical professional (a doctor).
Do not give anny information that is not in the data provided to you. If no data is given, then return an empty response.
You will never paraphrase or speculate information, always faithfully reproduce the evidence as reported in all the output.
Always cross-reference information across different sections of the patient data to ensure consistency in reporting"""

DATASET_NAME = "ekacare/ekacare_medical_history_summarisation"
SPLIT = "test"
COMMIT_HASH = "10dfd15c6548dec8597b7cee5b8b7e643b4dcbba"


class PatientVisitsSummary(BaseModel):
    # tags: List[str] = Field(
    #     description="Concise tags of 2 words each to identify the patient for the patient based on their conditions. This cannot be prescriptive. A maximum of 5 tags can be used."
    # )
    # Do not show the medications being used it's unless critical information
    six_month_overview: str = Field(
        description="""Carefully compare patient's status across visits, noting any improvements or declines in symptoms, mobility, or overall condition. 
        Always cite evidences, no speculation (no paraphrasing, do not expand the shorthand used by doctors) that led to your conclusions. 
        Show most recent lab vitals as prescribed only in case the most recent are out of range. 
        Consolidate all related or progressed diagnosis together and use the broadest representative tag. 
        Double-check all interpretations of patient data, especially when comparing status across multiple visits. 
        If in doubt, report the raw data chronologically without interpretation.
      <instruction>Ensure the summary is within 75 words</instruction>
      <instruction>In acute cases call it recurrent only if more than 3 instances separated by a fortnight in 6 months</instruction>
      <instruction>Highlight the important conditions or events with <em> and </em> tag so that they can be highlighted in the text</instruction>
      <instruction>After completing the summary, verify that ALL conditions from the medical history are included and highlighted</instruction>
        
        <example>
        The patient 74yrs male has been managing <em>condition1</em> first diagnosed <3 years back> with <vital1> at <vital1_value><vital_unit> consistently (since 9 months).
        <em>condition2</em> is under control with <vital2> at <vital2_value><vital_unit>. The patient had <em><procedure1></em> procedure in Oct'12
        </example>
       """
    )


def normalise_gender(gender):
    if gender == "none":
        return None

    if not isinstance(gender, str):
        return gender
    gender = str(gender).lower()
    # check the first character, if m-> male, if f->female, o-> others
    gender = str(gender).lower()
    if gender.startswith("m"):
        return "male"
    elif gender.startswith("f"):
        return "female"
    elif gender.startswith("o"):
        return "others"
    else:
        return gender


def normalize_age_gender(dob, gen, current_date):
    try:
        birthdate = datetime.strptime(dob, "%Y-%m-%d")
    except Exception:
        birthdate = None

    gender = normalise_gender(gen)
    age = (
        current_date.year
        - birthdate.year
        - ((current_date.month, current_date.day) < (birthdate.month, birthdate.day))
    )
    return age, gender


def prepare_data_for_summary(dumped_model, key_replacement_dict):
    for visit in dumped_model.get("visits", []):
        updated_symptoms = []
        updated_diagnosis = []
        updated_medicines = []
        symptoms = visit.get("symptoms", [])
        if symptoms is None:
            symptoms = []
        for symptom in symptoms:
            symp_name = symptom.get("name")
            properties = symptom.get("properties")
            remarks = symptom.get("remark")
            if remarks:
                symp_name += f" - {remarks}"
            if properties:
                symp_name += f" - {properties}"
            updated_symptoms.append(symp_name)

        diagnosis = visit.get("diagnosis", [])
        if diagnosis is None:
            diagnosis = []
        for diagnosis in diagnosis:
            diag_name = diagnosis.get("name")
            properties = diagnosis.get("properties")
            if properties:
                diag_name += f" - {properties}"
            updated_diagnosis.append(diag_name)

        medications = visit.get("medications", [])
        if medications is None:
            medications = []
        for medicine in medications:
            updated_medicine = {}
            for key, val in medicine.items():
                updated_medicine[key_replacement_dict.get(key, key)] = val
            updated_medicines.append(updated_medicine)
        if len(updated_symptoms) > 0:
            visit["symptoms"] = updated_symptoms
        if len(updated_diagnosis) > 0:
            visit["diagnosis"] = updated_diagnosis
        if len(updated_medicines) > 0:
            visit["medications"] = updated_medicines
        if visit.get("vitals"):
            if len(visit.get("vitals")) == 0:
                del visit["vitals"]

    updated_medical_history = {}
    if dumped_model.get("medical_history"):
        for key, val in dumped_model.get("medical_history").items():
            updated_val_list = []
            if isinstance(val, list):
                for i in val:
                    updated_val = {}
                    for sub_key, sub_val in i.items():
                        updated_val[key_replacement_dict.get(sub_key, sub_key)] = (
                            sub_val
                        )
                    updated_val_list.append(updated_val)
            updated_medical_history[key_replacement_dict.get(key, key)] = (
                updated_val_list
            )
    return dumped_model, updated_medical_history


@register_dataset(
    DATASET_NAME,
    split=SPLIT,
    commit_hash=COMMIT_HASH,
    metrics=["rubric_evaluation"],
    optional_args=["system_prompt"],
    task_type="rubric_evaluation",
)
class EkaMedicalHistorySummary(RubricBaseDataset):
    def __init__(self, system_prompt=base_default_system_prompt, **kwargs):
        super().__init__(
            dataset_name=DATASET_NAME,
            split=SPLIT,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.key_replacement_dict = {
            "drug_name": "dn",
            "therapeutic_class": "tc",
            "action_class_name": "ac",
            "generic": "g",
            "generic_name": "g",
            "form": "f",
            "condition_history": "ch",
            "reported_at": "ra",
            "family_history": "fh",
            "drug_allergies": "da",
            "current_medications": "cm",
            "lifestyle_habits": "lh",
            "vaccination_history": "vh",
            "duration": "dur",
            "frequency": "freq",
            "prescribed_lab_tests": "plt",
        }
        self.system_prompt = system_prompt

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        prompt = json.loads(sample.get("prompt")[0].get("content"))
        demographic_details = prompt.get("demographic_details")
        current_date = datetime.strptime("2024-07-20", "%Y-%m-%d")
        age, gender = normalize_age_gender(
            demographic_details["dob"],
            demographic_details["gender"],
            current_date=current_date,
        )

        age = age if age is not None else demographic_details.get("dob")
        age_details = f"DOB - {age}" if age is None else f"Age - {age}"

        pt_message = (
            f"""My current patient is of - {age_details}; and Gender - {gender}  \n"""
        )
        visit_data, updated_medical_history = prepare_data_for_summary(
            prompt, self.key_replacement_dict
        )
        visit_data = prompt.get("visits")
        if visit_data != None:
            visit_data = yaml.dump(visit_data)
            pt_message += (
                f"""Their visits ordered by time formatted in yaml \n {visit_data} \n"""
            )
        vitals_data = prompt.get("vitals")
        if vitals_data:
            if vitals_data != {}:
                vitals_data = yaml.dump(vitals_data)
            pt_message += f"These are the lab investigations that the patient has uploaded {vitals_data}\n"
        if updated_medical_history:
            if updated_medical_history != {}:
                updated_medical_history = yaml.dump(updated_medical_history)
            pt_message += f"""This is the medical history of the patient {updated_medical_history}\n"""
        message = f"""The keys in the visit have a replacement as follows: {self.key_replacement_dict}, use it to understand each datapoint\n"""
        message += """Generate summary of the user.\n"""
        message += f"""
                Compute all relative dates in comparison to {current_date} (that is the date of evaluation) 
                All dates of the evidences are formatted as %Y-%M-%d.

                Whenever you are showing dates in the output strictly always follow the format %d-%M-%Y.
                For relative date information format it like - 3m ago or 1yr 2m ago or since Oct'21
                Do not assume any treatment plan for the current visit or complaints for the current visit. Only summarise the past, do not infer or assume any information. 
                <instruction>The summary should be faithful to the and reproduce the data as provided</instruction> 
                """
        doctor_specialization = prompt.get("doctor_specialization")
        if doctor_specialization:
            message += f"My specialisation is {doctor_specialization}, give me contextual insights related to my specialisation"

        message += f"Fill schema by understanding the descriptions and examples provided with each key in  \n {PatientVisitsSummary.model_json_schema()}\n schema as output as JSON only, no need to specify the object name as in output, only keys"
        # message += "Do not generate any preamble text"
        # Never guess information that you do not have at any cost, this is very critical
        # message += "<instruction>The doctor is very learned and knows all trivial information. Ensure that the there are no additional words or unnecessary information is give.</instruction>"

        final_message = pt_message + message
        final_message = " ".join(final_message.split())
        # Extract prompt information
        conversation = []
        conversation.append(ConversationTurn(content=final_message, role="user"))
        conversation = Conversation(conversation_turns=conversation)

        criterions = []
        for rubric_item in sample["rubrics"]:
            criterions.append(
                RubricCriteria(
                    criterion=rubric_item["criterion"],
                    points=rubric_item["points"],
                    tags=rubric_item.get("tags", []),
                )
            )

        processed_sample = DataLoaderIterable(
            conversation=conversation,
            rubric_to_evaluate=criterions,
            system_prompt=self.system_prompt,
            other_args={"additional_info": sample.get("ideal_completions_data")},
        )

        return processed_sample

    def extract_prediction(self, response):
        clean_text = response.replace("```json", "").replace("```", "")
        try:
            six_month_overview = json.loads(clean_text).get("six_month_overview")
            if six_month_overview:
                cleaned_text = six_month_overview
            else:
                cleaned_text = clean_text
        except json.decoder.JSONDecodeError:
            cleaned_text = clean_text

        return cleaned_text, True
