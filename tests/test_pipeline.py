import os
from src.pipelines.training_pipeline import run_pipeline

def test_pipeline_runs_successfully(capsys):
    """
    Tests that the pipeline runs from start to finish without errors, 
    prints the expected metrics, and saves the model file.
    """
    model_path = "models/model.pkl"
    
    if os.path.exists(model_path):
        os.remove(model_path)

    run_pipeline()
    
    # A. Check that it printed the correct metrics using pytest's capsys fixture
    captured = capsys.readouterr()
    assert "CV Accuracy Mean:" in captured.out, "Pipeline failed to print CV Mean"
    assert "Model Accuracy:" in captured.out, "Pipeline failed to print Model Accuracy"
    assert "Cross-Validation Scores:" in captured.out, "Pipeline failed to print CV Scores"

    # B. Check that it successfully saved the model to the disk
    assert os.path.exists(model_path), f"The pipeline should have created {model_path}"