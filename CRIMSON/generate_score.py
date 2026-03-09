import json
import csv
import os
import sys
import re
from openai import OpenAI
from CRIMSON.prompt_parts import build_prompt as _build_evaluation_prompt_fn

from transformers import pipeline
import torch

class CRIMSONScore:
    """
    Calculate CRIMSON scores using various models (GPT, etc.) to evaluate
    radiology report generation quality.
    """
    
    DEFAULT_HF_MODEL = "CRIMSONScore/medgemma-4b-it-crimson"

    def __init__(
        self, 
        api="hf",
        model_name=None,
        device=None,
        torch_dtype=None,
        batch_size=1,
    ):
        """
        Initialize CRIMSON scorer.
        
        Args:
            api: Model type to use ("hf", "huggingface", "openai").
                 Defaults to "hf".
            model_name: Specific model name / deployment name.
                       Defaults to "CRIMSONScore/medgemma-4b-it-crimson" for HF.
                       For OpenAI, defaults to "gpt-5.2".
            device: Device to use for HuggingFace models. Defaults to "cuda".
            torch_dtype: PyTorch dtype for model weights (e.g., torch.float16, torch.bfloat16)
                        If None, uses torch.float16
            batch_size: Number of samples processed per forward pass for HuggingFace models
                        (e.g. MedGemma). Values > 1 enable batched inference. Defaults to 1.
        """
        self.api = api
        self.batch_size = batch_size
        
        if api == "openai":
            self.model_name = model_name or "gpt-5.2"
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        elif api in ["huggingface", "hf"]:
            self.model_name = model_name or self.DEFAULT_HF_MODEL
            self.device = device or "cuda"
            self.torch_dtype = torch_dtype or torch.float16
            
            print(f"Loading HuggingFace model: {self.model_name}")
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
            )
            print(f"Model loaded.")
        else:
            raise ValueError(f"Unsupported model: {api}. Use 'openai', 'huggingface', or 'hf'.")
    
    
    def _chat_completion(self, prompt):
        """
        Make a chat completion request (similar to process_reports.py pattern).
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response text
        """
        if self.api == "openai":
            messages = [
                {"role": "system", "content": "You are an expert radiology evaluator that assesses the accuracy of radiology reports."},
                {"role": "user", "content": prompt}
            ]
            
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "seed": 42,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        
        elif self.api in ["huggingface", "hf"]:
            messages = [
                {"role": "system", "content": "You are an expert radiology evaluator that assesses the accuracy of radiology reports."},
                {"role": "user", "content": prompt + "\nPlease respond with valid JSON only."},
            ]
            outputs = self.pipe(
                messages,
                max_new_tokens=4096,
                temperature=0,
                batch_size=self.batch_size,
            )
            return outputs[0]['generated_text'][-1]['content']
    
    def _build_evaluation_prompt(
        self, 
        reference_findings, 
        predicted_findings,
        patient_context=None,
        include_guidelines=True,
    ):
        """
        Build the evaluation prompt for CRIMSON scoring.
        Delegates to prompt_parts.build_prompt for composable prompt assembly.
        
        Args:
            reference_findings: Ground truth findings (string or list of strings)
            predicted_findings: Predicted findings (string or list of strings)
            patient_context: Optional patient context (age, indication, etc.)
            include_guidelines: If False, disables significance examples,
                attribute severity guidelines, and context guidelines.
            
        Returns:
            Formatted prompt string
        """
        return _build_evaluation_prompt_fn(
            reference_findings,
            predicted_findings,
            patient_context=patient_context,
            include_significance_examples=include_guidelines,
            include_attribute_guidelines=include_guidelines,
            include_context_guidelines=include_guidelines,
        )
    
    def evaluate(
        self,
        reference_findings,
        predicted_findings,
        patient_context=None,
        include_guidelines=True,
    ):
        """
        Evaluate findings and calculate CRIMSON score.
        
        Args:
            reference_findings: Ground truth findings
            predicted_findings: Model-generated findings  
            patient_context: Optional patient context
            include_guidelines: If False, disables guidelines in the prompt
            
        Returns:
            Dictionary containing:
                - raw_evaluation: Raw model output with errors by category
                - error_counts: Counts of each error type
                - metrics: Individual metric components
                - crimson_score: Final CRIMSON score
        """
        # Get evaluation from model
        prompt = self._build_evaluation_prompt(
            reference_findings, 
            predicted_findings, 
            patient_context,
            include_guidelines=include_guidelines,
        )
        
        response = self._chat_completion(prompt)
        
        try:
            evaluation = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse model response as JSON: {e}\nResponse: {response}")
        
        # Calculate CRIMSON score
        crimson_result = self._calculate_crimson(evaluation)
        
        return crimson_result
    
    def _calculate_crimson(
        self,
        evaluation
    ):
        """
        Calculate CRIMSON score from evaluation results.
        
        Args:
            evaluation: Parsed evaluation with matched findings and errors
            
        Returns:
            Dictionary with scores and metrics
        """
        errors = evaluation.get("errors", {})
        matched = evaluation.get("matched_findings", [])
        reference_findings_list = evaluation.get("reference_findings", [])
        predicted_findings_list = evaluation.get("predicted_findings", [])
        
        # Weight mapping for clinical significance
        significance_weights = {
            "urgent": 1.0,
            "actionable_not_urgent": 0.5,
            "not_actionable_not_urgent": 0.25,
            "benign_expected": 0.0,
        }
        
        # Weight mapping for attribute error severity
        attribute_severity_weights = {
            "significant": 0.5,
            "negligible": 0.0,
        }
        
        def calculate_weighted_count(error_list, weights=significance_weights, key="clinical_significance"):
            """Calculate weighted count based on significance/severity."""
            return sum(weights[error[key]] for error in error_list)
        
        ref_weight_by_id = {
            ref["id"]: significance_weights[ref["clinical_significance"]]
            for ref in reference_findings_list
        }
        pred_weight_by_id = {
            pred["id"]: significance_weights[pred["clinical_significance"]]
            for pred in predicted_findings_list
        }
        
        E_false = sum(pred_weight_by_id.get(f_id, 0.0) for f_id in errors.get("false_findings", []))
        
        E_miss = sum(ref_weight_by_id.get(m_id, 0.0) for m_id in errors.get("missing_findings", []))
        
        attr_errors = errors.get("attribute_errors", [])
        n_location = sum(1 for e in attr_errors if "location" in e.get("error_types", []))
        n_severity = sum(1 for e in attr_errors if "severity" in e.get("error_types", []))
        n_descriptor = sum(1 for e in attr_errors if "descriptor" in e.get("error_types", []))
        n_measurement = sum(1 for e in attr_errors if "measurement" in e.get("error_types", []))
        n_certainty = sum(1 for e in attr_errors if "certainty" in e.get("error_types", []))
        n_unspecific = sum(1 for e in attr_errors if "unspecific" in e.get("error_types", []))
        n_overinterpretation = sum(1 for e in attr_errors if "overinterpretation" in e.get("error_types", []))
        n_temporal = sum(1 for e in attr_errors if "temporal" in e.get("error_types", []))
        
        attr_errors_by_ref_id = {}
        for err in attr_errors:
            ref_id = err["ref_id"]
            if ref_id not in attr_errors_by_ref_id:
                attr_errors_by_ref_id[ref_id] = []
            attr_errors_by_ref_id[ref_id].append(err)
        
        N_G = calculate_weighted_count(reference_findings_list)
        if N_G == 0 and not reference_findings_list:
            N_G = len(matched) + E_miss
        
        E_penalty = E_false
        
        # Weighted sum of matched findings with partial credit for attribute errors
        matched_ref_ids = set()
        correct = 0.0
        for m in matched:
            ref_id = m["ref_id"]
            if ref_id in matched_ref_ids:
                continue  # Already counted this reference finding
            matched_ref_ids.add(ref_id)
            base_weight = ref_weight_by_id.get(ref_id, 0.0)
            
            finding_attr_errors = attr_errors_by_ref_id.get(ref_id, [])
            
            if not finding_attr_errors:
                correct += base_weight
            else:
                sum_error_weights = sum(
                    attribute_severity_weights[err["severity"]] for err in finding_attr_errors
                )
                credit_factor = base_weight / (base_weight + sum_error_weights) if (base_weight + sum_error_weights) > 0 else 0.0
                correct += base_weight * credit_factor
        
        errors_more_than_correct = E_penalty - correct
        
        if N_G == 0:
            S = 1.0 if E_penalty == 0 and E_miss == 0 else -(E_penalty + E_miss + 1)
        else:
            S = (correct - E_penalty) / N_G
        
        if S >= 0:
            crimson = S
        else:
            if errors_more_than_correct > 0:
                crimson = -1 * errors_more_than_correct / (1 + errors_more_than_correct)
            else:
                crimson = 0
        
        return {
            "raw_evaluation": evaluation,
            "error_counts": {
                "false_findings": len(errors.get("false_findings", [])),
                "missing_findings": len(errors.get("missing_findings", [])),
                "attribute_errors": len(attr_errors),
                "location_errors": n_location,
                "severity_errors": n_severity,
                "descriptor_errors": n_descriptor,
                "measurement_errors": n_measurement,
                "certainty_errors": n_certainty,
                "unspecific_errors": n_unspecific,
                "overinterpretation_errors": n_overinterpretation,
                "temporal_errors": n_temporal
            },
            "weighted_error_counts": {
                "false_findings": E_false,
                "missing_findings": E_miss,
                "attribute_errors": calculate_weighted_count(attr_errors, attribute_severity_weights, "severity")
            },
            "metrics": {
                "N_G": N_G,  # Number of ground truth findings
                "E_penalty": E_penalty,
                "correct": correct,
                "errors_more_than_correct": errors_more_than_correct,
                "S": S
            },
            "crimson_score": round(crimson, 4)
        }