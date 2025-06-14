# LayoutLM Evaluation Framework

A comprehensive evaluation tool for LayoutLM form understanding models that compares model predictions against human annotator labels.

## Purpose

This framework evaluates LayoutLM model performance on form understanding tasks by:

- **Comparing model predictions against gold standard human annotations**
- **Supporting multiple annotators** (annotator1_label, annotator2_label) as ground truth
- **Handling flexible data formats** with configurable prediction columns
- **Providing comprehensive metrics** including token-level and page-level accuracy, F1 scores
- **Generating detailed results** for in-depth analysis

## Key Features

- **Multi-format support**: Works with both CSV and Excel (.xlsx) files
- **Configurable evaluation**: Choose which annotator labels to use as ground truth
- **Flexible prediction columns**: Handle different data structures (pred vs labels columns)
- **Rich metrics**: Token-level and page-level accuracy, F1 (macro/micro/weighted), confusion matrices
- **Detailed reporting**: Per-class and per-page results with visual summaries

## Usage

### Basic Usage

```bash
# Evaluate using annotator1_label as ground truth, pred column as predictions
python evaluate.py annotation_labels/ --save-detailed-results

# Evaluate against annotator2_label
python evaluate.py annotation_labels/ --gold-standard annotator2_label --save-detailed-results

# Use labels column as predictions instead of pred column
python evaluate.py annotation_labels/ --prediction-col labels --save-detailed-results
```

### Configuration Options

```bash
python evaluate.py [PREDICTIONS_DIR] [OPTIONS]
```

**Required:**
- `PREDICTIONS_DIR`: Directory containing Excel/CSV files with annotations

**Key Options:**
- `--gold-standard` (`-g`): Gold standard column (`annotator1_label` or `annotator2_label`) [default: annotator1_label]
- `--prediction-col` (`-p`): Prediction column (`pred` or `labels`) [default: pred]
- `--output-dir` (`-o`): Base results directory [default: ./results] 
  - **Note**: Annotator suffix is automatically added (e.g., `results_annotator1`)
- `--save-detailed-results` (`-d`): Save detailed per-class and per-page results

### Expected Data Format

Your annotation files should contain these columns:
- `bboxes`: Bounding box coordinates
- `prob`: Prediction confidence scores
- `annotator1_label`: First annotator's labels (gold standard option 1)
- `annotator2_label`: Second annotator's labels (gold standard option 2)
- `pred` or `labels`: Model predictions (configurable)

## Example Workflows

### Compare Model Performance Against Different Annotators

```bash
# Evaluate against annotator 1 (creates results_evaluation_annotator1/)
python evaluate.py annotation_labels/ -g annotator1_label -o results_evaluation -d

# Evaluate against annotator 2 (creates results_evaluation_annotator2/)
python evaluate.py annotation_labels/ -g annotator2_label -o results_evaluation -d

# The annotator suffix is automatically added to directory names for easy comparison
```

### Evaluate Different Model Outputs

```bash
# If predictions are in 'pred' column
python evaluate.py annotation_labels/ -p pred -d

# If predictions are in 'labels' column  
python evaluate.py annotation_labels/ -p labels -d
```

## Output Files

The evaluation generates several output files in the results directory:

### Data Files
- **`summary_metrics.json`**: Overall evaluation metrics
- **`classification_report.csv`**: Per-class precision, recall, F1 scores
- **`confusion_matrix.csv`**: Confusion matrix for detailed error analysis
- **`per_page_results.csv`**: Page-by-page performance breakdown
- **`eval_config.json`**: Configuration used for this evaluation

### Visualizations (when `--save-detailed-results` is used)
- **`confusion_matrix_heatmap.png`**: Visual confusion matrix showing which form field types are confused with each other
- **`per_class_performance.png`**: Four-panel chart showing precision, recall, F1-score, and support for each form field class

## Environment Setup

```bash
# Activate the conda environment
conda activate du

# Install required dependencies (if not already installed)
pip install typer rich scikit-learn pandas openpyxl
```

## Metrics Explained

- **Token-level metrics**: Accuracy and F1 scores computed across all individual tokens
- **Page-level metrics**: Average performance across complete pages
- **Perfect accuracy pages**: Number of pages with 100% token accuracy
- **F1 scores**: Macro (unweighted average), micro (overall), and weighted (by support) variants

## Visualizations

When using `--save-detailed-results`, the evaluation generates helpful visualizations:

### Confusion Matrix Heatmap
- **Purpose**: Identify systematic labeling errors between form field types
- **Use Case**: Spot patterns like "supplier_a_pgs" being confused with "payer_a_pgs"
- **Interpretation**: Darker blue squares indicate more frequent confusions; diagonal shows correct predictions

### Per-Class Performance Charts
- **Four panels**: Precision, Recall, F1-Score, and Support (sample count)
- **Sorted by F1-Score**: Classes with worst performance appear on the right
- **Use Case**: Identify which form field types need model improvement
- **Interpretation**: Classes with low precision have many false positives; low recall means many missed detections

## Use Cases

1. **Model Development**: Evaluate model improvements during training
2. **Annotator Agreement**: Compare performance against different human annotators
3. **Production Monitoring**: Assess model performance on real-world data
4. **Error Analysis**: Identify problematic label categories and pages
5. **Baseline Comparison**: Compare different model architectures or approaches