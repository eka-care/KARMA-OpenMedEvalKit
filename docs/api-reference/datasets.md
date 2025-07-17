# Datasets API Reference

This section documents KARMA's dataset system, including base classes, built-in datasets, and integration patterns.

## Base Classes

### BaseMultimodalDataset

The foundation for all multimodal evaluation datasets in KARMA.

::: karma.eval_datasets.base_dataset.BaseMultimodalDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Built-in Datasets

### Medical Question Answering

#### MedQADataset (openlifescienceai/medqa)

Medical Question Answering dataset.

::: karma.eval_datasets.medqa_dataset.MedQADataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### PubMedMCQADataset (openlifescienceai/pubmedqa)

PubMed Multiple Choice Question Answering dataset.

::: karma.eval_datasets.pubmedmcqa_dataset.PubMedMCQADataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MedMCQADataset (openlifescienceai/medmcqa)

Medical Multiple Choice Question Answering dataset.

::: karma.eval_datasets.medmcqa_dataset.MedMCQADataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MedXpertQADataset (ChuGyouk/MedXpertQA)

Medical Expert Question Answering dataset.

::: karma.eval_datasets.medxpertqa_dataset.MedXpertQADataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### Vision-Language Datasets

#### SLAKEDataset (mdwiratathya/SLAKE-vqa-english)

Structured Language And Knowledge Extraction dataset for medical VQA.

::: karma.eval_datasets.slake_dataset.SLAKEDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### VQARADDataset (flaviagiammarino/vqa-rad)

Visual Question Answering for Radiology dataset.

::: karma.eval_datasets.vqa_rad_dataset.VQARADDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### Language and Speech Datasets

#### IN22ConvDataset (ai4bharat/IN22-Conv)

Indic Language Conversation Translation dataset.

::: karma.eval_datasets.in22conv_dataset.IN22ConvDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### IndicVoicesRDataset (ai4bharat/indicvoices_r)

Indic Voices Recognition dataset for ASR evaluation.

::: karma.eval_datasets.indicvoices_r_dataset.IndicVoicesRDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### MMLU Medical Datasets

Medical benchmarks from the MMLU suite.

#### MMLUProfessionalMedicineDataset (openlifescienceai/mmlu_professional_medicine)

MMLU Professional Medicine dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUProfessionalMedicineDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MMLUAnatomyDataset (openlifescienceai/mmlu_anatomy)

MMLU Anatomy dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUAnatomyDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MMLUCollegeBiologyDataset (openlifescienceai/mmlu_college_biology)

MMLU College Biology dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUCollegeBiologyDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MMLUClinicalKnowledgeDataset (openlifescienceai/mmlu_clinical_knowledge)

MMLU Clinical Knowledge dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUClinicalKnowledgeDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### MMLUCollegeMedicineDataset (openlifescienceai/mmlu_college_medicine)

MMLU College Medicine dataset.

::: karma.eval_datasets.mmlu_medical_datasets.MMLUCollegeMedicineDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

### Rubric-Based Evaluation Datasets

#### RubricBaseDataset

Base class for rubric-based evaluation datasets that handle medical question answering with rubric-based evaluation.

::: karma.eval_datasets.rubrics.rubric_base_dataset.RubricBaseDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### EkaMedicalHistorySummary (ekacare/ekacare_medical_history_summarisation)

EkaCare Medical History Summarization dataset for rubric-based evaluation.

::: karma.eval_datasets.rubrics.eka_medical_history_summary.EkaMedicalHistorySummary
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

#### HealthBenchDataset (Tonic/Health-Bench-Eval-OSS-2025-07)

Health-Bench evaluation dataset for rubric-based medical question answering.

::: karma.eval_datasets.rubrics.healthbench_dataset.HealthBenchDataset
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true

## Data Models

### DataLoaderIterable

Pydantic model for multimodal dataset samples.

::: karma.data_models.dataloader_iterable.DataLoaderIterable
    options:
      show_source: false
      show_root_heading: true
      show_category_heading: true
      members_order: source
      filters: ["!^_"]
      merge_init_into_class: true
