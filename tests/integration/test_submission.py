import json
import os
from datetime import datetime

import gradio
import pytest
import pyarrow as pa
from agenteval.models import SubmissionMetadata
from datasets import load_dataset, VerificationMode
from huggingface_hub import HfApi, hf_hub_download

from aliases import CANONICAL_TOOL_USAGE_STANDARD, CANONICAL_OPENNESS_CLOSED_UI_ONLY
from config import IS_INTERNAL, CONFIG_NAME, CONTACT_DATASET, SUBMISSION_DATASET
from submission import add_new_eval

_hf = HfApi()


class TestSubmission:
    @pytest.fixture(autouse=True)
    def setup(self):
        # These need to be set before imports are evaluated so all we can do here
        # is check that they have been set correctly.
        assert IS_INTERNAL == True
        assert CONFIG_NAME == "continuous-integration"

    def test_add_new_eval(self, mocker):
        # Bypass some checks so that the test can cover later parts of the code.
        mocker.patch("submission._is_hf_acct_too_new", return_value=False)
        mocker.patch("submission._is_last_submission_too_recent", return_value=False)

        # We use this to find records corresponding to this test.
        agent_description = f"CI run at {datetime.now().isoformat()}"
        print(f"Using unique agent description: {agent_description}")

        print("Submitting test submission...")
        with open(os.path.join(os.path.dirname(__file__), "test-submission.tar.gz"), "rb") as f:
            result = add_new_eval(
                val_or_test="test",
                agent_name="TestSubmissionIntegration",
                agent_description=agent_description,
                agent_url="https://github.com/allenai/asta-bench-leaderboard/blob/main/tests/integration/test_submission.py",
                openness=CANONICAL_OPENNESS_CLOSED_UI_ONLY,
                degree_of_control=CANONICAL_TOOL_USAGE_STANDARD,
                path_to_file=f,
                username="test_user",
                role="Other",
                email="jasond+asta_testing@allenai.org",
                email_opt_in=True,
                profile=gradio.OAuthProfile({
                    "name": "Test User",
                    "preferred_username": "test_user",
                    "profile": "test_user_profile",
                    "picture": "https://placecats.com/150/150",
                }),
            )

        message, error_modal, success_modal, loading_modal = result
        assert message == ""  # Success
        assert error_modal == {'__type__': 'update', 'visible': False}
        assert success_modal == {'__type__': 'update', 'visible': True}
        assert loading_modal == {'__type__': 'update', 'visible': False}

        print("Looking up contact record...")
        contacts = load_dataset(path=CONTACT_DATASET,
                                name=CONFIG_NAME,
                                download_mode="force_redownload",
                                verification_mode=VerificationMode.NO_CHECKS)
        # There should have been a new entry due to this test with our unique description.
        found_contact = next(row for row in contacts['test'] if row['agent_description'] == agent_description)
        assert found_contact

        # This contains an attribute that should lead us to files in the submissions dataset.
        dataset_url = found_contact['dataset_url']
        print(f"Found dataset URL: {dataset_url}")
        assert dataset_url.startswith(
            "hf://datasets/allenai/asta-bench-internal-submissions/continuous-integration/test/")

        print("Checking submission dataset...")
        # Commit message itself should link this and the contact record together unambiguously.
        recent_commits = _hf.list_repo_commits(repo_type="dataset", repo_id=SUBMISSION_DATASET)
        assert any(dataset_url in c.title for c in recent_commits)

        print("Checking that files are present...")
        rel_path = dataset_url[len("hf://datasets/allenai/asta-bench-internal-submissions/"):]
        ds_info = _hf.dataset_info(SUBMISSION_DATASET)
        # These are the files in our test-submission.tar.gz
        assert any(f"{rel_path}/eval_config.json" == f.rfilename for f in ds_info.siblings)
        assert any(f"{rel_path}/task_sqa_solver_openscilm.eval" == f.rfilename for f in ds_info.siblings)
        # This is the generated metadata put into the dataset itself.
        assert any(f"{rel_path}/submission.json" == f.rfilename for f in ds_info.siblings)

        print("Checking contact record against submission.json...")
        # Checks on contact record which is stored in a private dataset.
        local_path = hf_hub_download(repo_type="dataset",
                                     repo_id=SUBMISSION_DATASET,
                                     filename=f"{rel_path}/submission.json")
        with open(local_path) as f:
            contact_from_json = json.load(f)
        # Assert that all keys and values in submission.json are present in the contact record
        for key, value_from_json in contact_from_json.items():
            value_from_dataset = found_contact[key]
            if isinstance(value_from_dataset, datetime):
                value_from_dataset = found_contact[key].isoformat().replace('+00:00', 'Z')
            assert value_from_dataset == value_from_json
        # submission.json should not contain sensitive PII, specifically, email.
        assert 'email' in found_contact
        assert 'email' not in contact_from_json
        # submission.json is defined by a specific data model.
        SubmissionMetadata.model_validate(contact_from_json)
