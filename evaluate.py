"""Main evaluation script for LayoutLM form understanding."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass
class RichConfig:
    """Configuration for rich console output."""

    console: Console = Console()
    success_style: str = "[bold green]\u2705[/bold green]"
    fail_style: str = "[bold red]\u274C[/bold red]"
    warning_style: str = "[bold yellow]\u26A0[/bold yellow]"
    info_style: str = "[bold blue]ℹ[/bold blue]"


rich_config = RichConfig()


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("evaluation.log"),
        ],
    )


def load_prediction_files(predictions_dir: Path, gold_standard_col: str, prediction_col: str) -> List[pd.DataFrame]:
    """Load all CSV or Excel prediction files from directory.

    Args:
        predictions_dir: Directory containing CSV or Excel prediction files
        gold_standard_col: Column name for gold standard labels (e.g., 'annotator1_label')
        prediction_col: Column name for predictions (e.g., 'pred' or 'labels')

    Returns:
        List of DataFrames, one per page
    """
    # Look for both CSV and Excel files
    csv_files = list(predictions_dir.glob("*.csv"))
    excel_files = list(predictions_dir.glob("*.xlsx"))
    
    all_files = csv_files + excel_files
    if not all_files:
        raise ValueError(f"No CSV or Excel files found in {predictions_dir}")

    dataframes = []
    processed_files = []
    skipped_files = []
    
    for file_path in sorted(all_files):
        # Load based on file extension
        if file_path.suffix.lower() == '.csv':
            page_df = pd.read_csv(file_path)
            processed_files.append(file_path)
        else:  # Excel file
            file_loaded = False
            
            # Try exception handling approach to read all data before corruption
            def try_minimal_columns(filepath):
                import openpyxl
                wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
                ws = wb.active
                data = []
                headers = []
                row_num = 0
                
                try:
                    for row in ws.iter_rows(values_only=True):
                        if row_num == 0:
                            headers = [str(cell) if cell is not None else f"col_{j}" for j, cell in enumerate(row)]
                        else:
                            data.append(row)
                        row_num += 1
                        
                        # Safety limit to prevent infinite loops
                        if row_num > 1000:
                            break
                            
                except Exception:
                    # Stop gracefully when corruption is detected
                    pass
                
                wb.close()
                return pd.DataFrame(data, columns=headers)
            
            approaches = [
                ("minimal column extraction", lambda filepath=file_path: try_minimal_columns(filepath)),
                ("pandas default read-only", lambda filepath=file_path: pd.read_excel(filepath)),
                ("openpyxl read-only", lambda filepath=file_path: pd.read_excel(filepath, engine='openpyxl'))
            ]
            
            for approach_name, read_func in approaches:
                try:
                    page_df = read_func()
                    if page_df is not None and not page_df.empty:
                        processed_files.append(file_path)
                        file_loaded = True
                        rich_config.console.print(f"{rich_config.success_style} Loaded {file_path.name} using {approach_name}")
                        break
                except Exception as e:
                    rich_config.console.print(f"{rich_config.info_style} {approach_name} failed for {file_path.name}: {str(e)[:100]}")
                    continue
            
            if not file_loaded:
                rich_config.console.print(f"{rich_config.fail_style} All approaches failed for {file_path.name}")
                skipped_files.append(file_path)
                continue
            
        # Validate required columns
        required_cols = ["bboxes", prediction_col, "prob", gold_standard_col]
        missing_cols = [col for col in required_cols if col not in page_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
        
        # Standardize column names for evaluation
        page_df["labels"] = page_df[gold_standard_col]  # Gold standard
        page_df["pred"] = page_df[prediction_col]       # Predictions
        dataframes.append(page_df)

    # Display processing summary
    rich_config.console.print(f"{rich_config.success_style} Processed {len(processed_files)} files successfully")
    rich_config.console.print(f"{rich_config.info_style} Skipped {len(skipped_files)} corrupted files")
    
    if skipped_files:
        rich_config.console.print("\n[bold]Skipped files:[/bold]")
        for file_path in skipped_files:
            rich_config.console.print(f"  - {file_path.name}")

    return dataframes


def compute_token_metrics(all_predictions: np.ndarray, all_labels: np.ndarray) -> Dict:
    """Compute token-level evaluation metrics.

    Args:
        all_predictions: Array of predicted class labels
        all_labels: Array of ground truth class labels

    Returns:
        Dictionary containing token-level metrics
    """
    # Convert all labels to strings for consistency (since we have mix of string and numeric labels)
    all_predictions_str = [str(pred) for pred in all_predictions]
    all_labels_str = [str(label) for label in all_labels]
    
    accuracy = accuracy_score(all_labels_str, all_predictions_str)
    f1_macro = f1_score(all_labels_str, all_predictions_str, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels_str, all_predictions_str, average="micro", zero_division=0)
    f1_weighted = f1_score(
        all_labels_str, all_predictions_str, average="weighted", zero_division=0
    )

    # Classification report for per-class metrics
    class_report = classification_report(
        all_labels_str, all_predictions_str, output_dict=True, zero_division=0
    )

    return {
        "token_accuracy": accuracy,
        "token_f1_macro": f1_macro,
        "token_f1_micro": f1_micro,
        "token_f1_weighted": f1_weighted,
        "classification_report": class_report,
        "confusion_matrix": confusion_matrix(all_labels_str, all_predictions_str).tolist(),
    }


def compute_page_metrics(page_dataframes: List[pd.DataFrame]) -> Dict:
    """Compute page-level evaluation metrics.

    Args:
        page_dataframes: List of DataFrames, one per page

    Returns:
        Dictionary containing page-level metrics
    """
    page_accuracies = []
    page_f1_scores = []
    perfect_accuracy_pages = 0

    for page_df in page_dataframes:
        if len(page_df) == 0:
            continue

        # Convert to strings for consistency
        page_labels_str = [str(label) for label in page_df["labels"]]
        page_pred_str = [str(pred) for pred in page_df["pred"]]
        
        page_acc = accuracy_score(page_labels_str, page_pred_str)
        page_f1 = f1_score(
            page_labels_str, page_pred_str, average="macro", zero_division=0
        )

        page_accuracies.append(page_acc)
        page_f1_scores.append(page_f1)

        if page_acc == 1.0:
            perfect_accuracy_pages += 1

    return {
        "page_accuracy_mean": np.mean(page_accuracies) if page_accuracies else 0,
        "page_accuracy_std": np.std(page_accuracies) if page_accuracies else 0,
        "page_f1_mean": np.mean(page_f1_scores) if page_f1_scores else 0,
        "page_f1_std": np.std(page_f1_scores) if page_f1_scores else 0,
        "pages_perfect_accuracy": perfect_accuracy_pages,
        "total_pages": len(
            [page_df for page_df in page_dataframes if len(page_df) > 0]
        ),
    }


def create_confusion_matrix_heatmap(confusion_mat: np.ndarray, class_labels: List[str], output_path: Path) -> None:
    """Create and save confusion matrix heatmap visualization.
    
    Args:
        confusion_mat: Confusion matrix as numpy array
        class_labels: List of class label names
        output_path: Path to save the heatmap image
    """
    plt.figure(figsize=(15, 12))
    
    # Create heatmap with better formatting
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Number of Predictions'}
    )
    
    plt.title('Confusion Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    plt.ylabel('True Labels', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_per_class_performance_chart(class_report: Dict, output_path: Path) -> None:
    """Create and save per-class performance bar chart.
    
    Args:
        class_report: Classification report dictionary from sklearn
        output_path: Path to save the chart image
    """
    # Extract per-class metrics (exclude summary statistics)
    class_metrics = {}
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
            class_metrics[class_name] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            }
    
    if not class_metrics:
        return
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(class_metrics).T
    
    # Create subplots for different metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Sort classes by F1-score for better visualization
    df_sorted = metrics_df.sort_values('f1-score', ascending=False)
    
    # Precision bar chart
    bars1 = ax1.bar(range(len(df_sorted)), df_sorted['precision'], color='skyblue', alpha=0.8)
    ax1.set_title('Precision by Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Precision', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted.index, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Recall bar chart
    bars2 = ax2.bar(range(len(df_sorted)), df_sorted['recall'], color='lightcoral', alpha=0.8)
    ax2.set_title('Recall by Class', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(len(df_sorted)))
    ax2.set_xticklabels(df_sorted.index, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # F1-score bar chart
    bars3 = ax3.bar(range(len(df_sorted)), df_sorted['f1-score'], color='lightgreen', alpha=0.8)
    ax3.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.set_xticks(range(len(df_sorted)))
    ax3.set_xticklabels(df_sorted.index, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Support (sample count) bar chart
    bars4 = ax4.bar(range(len(df_sorted)), df_sorted['support'], color='gold', alpha=0.8)
    ax4.set_title('Support (Sample Count) by Class', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Samples', fontweight='bold')
    ax4.set_xticks(range(len(df_sorted)))
    ax4.set_xticklabels(df_sorted.index, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax = bar.axes
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(df_sorted['support'])*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def configure_evaluation(
    predictions_dir: str,
    output_dir: str,
    gold_standard_col: str,
    prediction_col: str,
    num_classes: int,
    class_names_file: Optional[str],
    log_level: str,
    save_detailed_results: bool,
) -> Dict:
    """Extract and display configuration for evaluation."""
    console = rich_config.console

    # Display configuration
    config_table = Table(title="Evaluation Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Predictions Directory", predictions_dir)
    config_table.add_row("Output Directory", output_dir)
    config_table.add_row("Gold Standard Column", gold_standard_col)
    config_table.add_row("Prediction Column", prediction_col)
    config_table.add_row("Number of Classes", str(num_classes))
    config_table.add_row("Class Names File", class_names_file or "None")
    config_table.add_row("Log Level", log_level)
    config_table.add_row("Save Detailed Results", str(save_detailed_results))

    console.print(config_table)

    return {
        "predictions_dir": predictions_dir,
        "output_dir": output_dir,
        "gold_standard_col": gold_standard_col,
        "prediction_col": prediction_col,
        "num_classes": num_classes,
        "class_names_file": class_names_file,
        "log_level": log_level,
        "save_detailed_results": save_detailed_results,
    }


app = typer.Typer(help="Evaluate LayoutLM predictions from CSV files")


@app.command()
def main(
    predictions_dir: str = typer.Argument(
        ..., help="Directory containing CSV or Excel prediction files (one per page)"
    ),
    output_dir: str = typer.Option(
        "./results", "--output-dir", "-o", help="Directory to save evaluation results"
    ),
    gold_standard_col: str = typer.Option(
        "annotator1_label", "--gold-standard", "-g", help="Column name for gold standard labels (annotator1_label or annotator2_label)"
    ),
    prediction_col: str = typer.Option(
        "pred", "--prediction-col", "-p", help="Column name for predictions (pred or labels)"
    ),
    num_classes: int = typer.Option(
        58, "--num-classes", "-n", help="Number of classification categories"
    ),
    class_names_file: Optional[str] = typer.Option(
        None, "--class-names-file", "-c", help="Path to file containing class names"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level", case_sensitive=False
    ),
    save_detailed_results: bool = typer.Option(
        False,
        "--save-detailed-results",
        "-d",
        help="Save detailed per-class and per-page results",
    ),
) -> None:
    """Main evaluation function."""
    console = rich_config.console

    # Configure and display settings
    config = configure_evaluation(
        predictions_dir,
        output_dir,
        gold_standard_col,
        prediction_col,
        num_classes,
        class_names_file,
        log_level,
        save_detailed_results,
    )

    # Setup logging
    setup_logging(log_level)

    # Create output directory with annotator suffix
    base_output_dir = Path(output_dir)
    annotator_suffix = gold_standard_col.replace("_label", "")  # e.g., "annotator1" from "annotator1_label"
    
    # If the base directory name doesn't already include annotator info, add it
    if annotator_suffix not in base_output_dir.name:
        output_dir_path = base_output_dir.parent / f"{base_output_dir.name}_{annotator_suffix}"
    else:
        output_dir_path = base_output_dir
        
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Save evaluation configuration
    config_path = output_dir_path / "eval_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # Load class names if provided (currently unused but kept for future extension)
    if class_names_file:
        class_names_path = Path(class_names_file)
        class_names_path.read_text().strip().split("\n")

    # Load prediction files
    console.print(
        Panel(f"{rich_config.info_style} Loading prediction files from {predictions_dir}")
    )
    predictions_dir_path = Path(predictions_dir)
    page_dataframes = load_prediction_files(predictions_dir_path, gold_standard_col, prediction_col)
    console.print(
        f"{rich_config.success_style} Loaded [bold green]{len(page_dataframes)}[/bold green] prediction files"
    )

    # Combine all predictions and labels for token-level metrics
    all_predictions = []
    all_labels = []
    total_tokens = 0

    for page_df in page_dataframes:
        if len(page_df) > 0:
            all_predictions.extend(page_df["pred"].tolist())
            all_labels.extend(page_df["labels"].tolist())
            total_tokens += len(page_df)

    console.print(
        f"Total tokens across all pages: [bold cyan]{total_tokens}[/bold cyan]"
    )

    # Compute token-level metrics
    with console.status("[bold yellow]Computing token-level metrics...[/bold yellow]"):
        token_metrics = compute_token_metrics(
            np.array(all_predictions), np.array(all_labels)
        )
    console.print(f"{rich_config.success_style} Token-level metrics computed")

    # Compute page-level metrics
    with console.status("[bold yellow]Computing page-level metrics...[/bold yellow]"):
        page_metrics = compute_page_metrics(page_dataframes)
    console.print(f"{rich_config.success_style} Page-level metrics computed")

    # Combine all metrics
    final_metrics = {
        **token_metrics,
        **page_metrics,
        "total_tokens": total_tokens,
        "avg_tokens_per_page": total_tokens / page_metrics["total_pages"]
        if page_metrics["total_pages"] > 0
        else 0,
    }

    # Save detailed results if requested
    if save_detailed_results:
        console.print(f"{rich_config.info_style} Saving detailed results...")

        # Save classification report
        class_report_df = pd.DataFrame(
            token_metrics["classification_report"]
        ).transpose()
        class_report_df.to_csv(output_dir_path / "classification_report.csv")

        # Save confusion matrix
        confusion_df = pd.DataFrame(token_metrics["confusion_matrix"])
        confusion_df.to_csv(output_dir_path / "confusion_matrix.csv", index=False)

        # Save per-page results
        page_results = []
        for i, page_df in enumerate(page_dataframes):
            if len(page_df) > 0:
                page_acc = accuracy_score(page_df["labels"], page_df["pred"])
                page_f1 = f1_score(
                    page_df["labels"], page_df["pred"], average="macro", zero_division=0
                )
                page_results.append(
                    {
                        "page_id": i,
                        "num_tokens": len(page_df),
                        "accuracy": page_acc,
                        "f1_macro": page_f1,
                    }
                )

        page_results_df = pd.DataFrame(page_results)
        page_results_df.to_csv(output_dir_path / "per_page_results.csv", index=False)

        console.print(f"{rich_config.success_style} Detailed results saved")

    # Generate visualizations if detailed results are requested
    if save_detailed_results:
        console.print(f"{rich_config.info_style} Generating visualizations...")
        
        # Get unique class labels for confusion matrix
        all_unique_labels = list(set(all_labels + all_predictions))
        
        # Create confusion matrix heatmap
        conf_matrix_path = output_dir_path / "confusion_matrix_heatmap.png"
        create_confusion_matrix_heatmap(
            np.array(token_metrics["confusion_matrix"]), 
            all_unique_labels, 
            conf_matrix_path
        )
        
        # Create per-class performance chart
        performance_chart_path = output_dir_path / "per_class_performance.png"
        create_per_class_performance_chart(
            token_metrics["classification_report"],
            performance_chart_path
        )
        
        console.print(f"{rich_config.success_style} Visualizations generated")

    # Save summary metrics
    # Remove non-serializable items for JSON
    json_metrics = {
        k: v
        for k, v in final_metrics.items()
        if k not in ["classification_report", "confusion_matrix"]
    }

    summary_path = output_dir_path / "summary_metrics.json"
    summary_path.write_text(json.dumps(json_metrics, indent=2))

    # Create summary table
    summary_table = Table(title="LAYOUTLM EVALUATION SUMMARY", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    # Overview metrics
    summary_table.add_row(
        "Total tokens evaluated", f"{final_metrics.get('total_tokens', 0):,}"
    )
    summary_table.add_row(
        "Total pages evaluated", f"{final_metrics.get('total_pages', 0):,}"
    )
    summary_table.add_row(
        "Average tokens per page", f"{final_metrics.get('avg_tokens_per_page', 0):.1f}"
    )
    summary_table.add_row("", "")

    # Token-level metrics
    summary_table.add_row("[bold]TOKEN-LEVEL METRICS[/bold]", "")
    summary_table.add_row("  Accuracy", f"{final_metrics.get('token_accuracy', 0):.4f}")
    summary_table.add_row(
        "  F1 (macro)", f"{final_metrics.get('token_f1_macro', 0):.4f}"
    )
    summary_table.add_row(
        "  F1 (micro)", f"{final_metrics.get('token_f1_micro', 0):.4f}"
    )
    summary_table.add_row(
        "  F1 (weighted)", f"{final_metrics.get('token_f1_weighted', 0):.4f}"
    )
    summary_table.add_row("", "")

    # Page-level metrics
    summary_table.add_row("[bold]PAGE-LEVEL METRICS[/bold]", "")
    summary_table.add_row(
        "  Accuracy (mean)", f"{final_metrics.get('page_accuracy_mean', 0):.4f}"
    )
    summary_table.add_row(
        "  Accuracy (std)", f"{final_metrics.get('page_accuracy_std', 0):.4f}"
    )
    summary_table.add_row("  F1 (mean)", f"{final_metrics.get('page_f1_mean', 0):.4f}")
    summary_table.add_row("  F1 (std)", f"{final_metrics.get('page_f1_std', 0):.4f}")
    summary_table.add_row(
        "  Perfect accuracy pages", f"{final_metrics.get('pages_perfect_accuracy', 0)}"
    )

    console.print("\n")
    console.print(summary_table)
    console.print("\n")

    console.print(
        Panel(
            f"{rich_config.success_style} Evaluation completed successfully!\n"
            f"Results saved to: [bold blue]{output_dir_path}[/bold blue]",
            style="green",
        )
    )


if __name__ == "__main__":
    app()
