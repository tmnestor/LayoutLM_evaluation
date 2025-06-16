# LayoutLM Evaluation Tool

This tool evaluates LayoutLM model predictions for form understanding tasks, handling both CSV and Excel prediction files.

## F1 Score Metrics Explained

When evaluating named entity recognition models like LayoutLM, understanding the different F1 metrics is crucial, especially with imbalanced datasets where "other" labels dominate.

### F1 Macro
- **Calculation**: Computes F1 for each class separately, then takes the unweighted average
- **Use case**: Best for understanding per-class performance across all entity types
- **Characteristics**: Treats rare classes (e.g., "supplier_name" with 5 examples) equally with frequent classes (e.g., "other" with 5000 examples)
- **Bias**: Heavily influenced by performance on rare classes
- **When to use**: When you care about detecting specific named entities regardless of their frequency

### F1 Micro
- **Calculation**: Pools all true positives, false positives, and false negatives across all classes, then computes F1
- **Use case**: Overall system performance across all predictions
- **Characteristics**: Essentially becomes overall accuracy for classification tasks
- **Bias**: Heavily influenced by performance on frequent classes ("other")
- **When to use**: When you want to measure overall prediction accuracy

### F1 Weighted
- **Calculation**: Computes F1 for each class, then takes support-weighted average (weighted by number of examples)
- **Use case**: Balance between macro and micro metrics
- **Characteristics**: Frequent classes get higher weight but rare class performance is still considered
- **Bias**: More influenced by frequent classes but accounts for rare class performance
- **When to use**: When you want a realistic performance measure that considers class distribution

## Recommendations for Named Entity Recognition

For LayoutLM form understanding tasks:

- **Primary metric**: **F1 Macro** - Most important for entity extraction tasks since it shows how well the model performs on the actual named entity labels you care about
- **Secondary metric**: **F1 Weighted** - Good middle ground providing realistic performance assessment
- **Less important**: **F1 Micro** - Can be misleading as it will be artificially high due to correct "other" predictions

### Example Scenario
If your model correctly identifies:
- 95% of "other" tokens (5000 examples)  
- 60% of "supplier_name" tokens (10 examples)
- 70% of "address" tokens (20 examples)

Results would be:
- **F1 Micro**: ~94% (dominated by "other" performance)
- **F1 Weighted**: ~92% (weighted toward "other" but considers entity performance)  
- **F1 Macro**: ~75% (average of all three classes equally weighted)

The F1 Macro score of 75% gives the most realistic view of named entity extraction performance.

## Usage

```bash
python evaluate.py /path/to/predictions --gold-standard annotator1_label --prediction-col pred
```

The tool automatically handles:
- Excel files with corrupted metadata
- Extreme class imbalance by filtering "other"-only rows
- None value handling
- Multiple file formats (CSV and Excel)