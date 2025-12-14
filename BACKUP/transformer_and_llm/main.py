"""
Assignment 4: Transformer and LLMs
Main entry point that runs all sections of the assignment.

Sections:
1. Transformer Classifier for Sentiment Analysis
2. Text Summarization (Extractive and Abstractive)
3. Parameter-Efficient Fine-tuning
4. Retrieval-Augmented Generation (RAG)

Usage:
    python main.py           # Run all sections
    python main.py --part 1  # Run only Part 1
    python main.py --part 2  # Run only Part 2
    python main.py --part 3  # Run only Part 3
    python main.py --part 4  # Run only Part 4
"""

import sys
import argparse


def run_part1():
    """Run Part 1: Transformer Classifier for Sentiment Analysis."""
    print("\n" + "="*70)
    print("PART 1: TRANSFORMER CLASSIFIER FOR SENTIMENT ANALYSIS")
    print("="*70 + "\n")
    
    try:
        from transformer_classifier import main as transformer_main
        transformer_main()
    except Exception as e:
        print(f"Error in Part 1: {e}")
        import traceback
        traceback.print_exc()


def run_part2():
    """Run Part 2: Text Summarization (Extractive and Abstractive)."""
    print("\n" + "="*70)
    print("PART 2: TEXT SUMMARIZATION")
    print("="*70 + "\n")
    
    # Part 2.1: Extractive Summarization
    print("-" * 50)
    print("Part 2.1: Extractive Summarization")
    print("-" * 50)
    
    try:
        from extractive_summary import main as extractive_main
        extractive_main()
    except Exception as e:
        print(f"Error in Part 2.1: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")
    
    # Part 2.2: Abstractive Summarization
    print("-" * 50)
    print("Part 2.2: Abstractive Summarization")
    print("-" * 50)
    
    try:
        from abstractive_summary import main as abstractive_main
        abstractive_main()
    except Exception as e:
        print(f"Error in Part 2.2: {e}")
        import traceback
        traceback.print_exc()


def run_part3():
    """Run Part 3: Parameter-Efficient Fine-tuning."""
    print("\n" + "="*70)
    print("PART 3: PARAMETER-EFFICIENT FINE-TUNING")
    print("="*70 + "\n")
    
    try:
        from params_finetune import main as finetune_main
        finetune_main()
    except Exception as e:
        print(f"Error in Part 3: {e}")
        import traceback
        traceback.print_exc()


def run_part4():
    """Run Part 4: Retrieval-Augmented Generation (RAG)."""
    print("\n" + "="*70)
    print("PART 4: RETRIEVAL-AUGMENTED GENERATION (RAG)")
    print("="*70 + "\n")
    
    try:
        from rag_tune import main as rag_main
        rag_main()
    except Exception as e:
        print(f"Error in Part 4: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Assignment 4: Transformer and LLMs"
    )
    parser.add_argument(
        "--part", 
        type=int, 
        choices=[1, 2, 3, 4],
        help="Run only a specific part (1-4). If not specified, runs all parts."
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ASSIGNMENT 4: TRANSFORMER AND LLMs")
    print("="*70)
    print()
    print("This assignment covers:")
    print("  1. Sequence Classification using Transformer Encoder")
    print("  2. Extractive and Abstractive Text Summarization")
    print("  3. Parameter-Efficient Fine-tuning of Pretrained LLM")
    print("  4. Retrieval-Augmented Generation (RAG)")
    print()
    
    if args.part:
        # Run specific part
        if args.part == 1:
            run_part1()
        elif args.part == 2:
            run_part2()
        elif args.part == 3:
            run_part3()
        elif args.part == 4:
            run_part4()
    else:
        # Run all parts
        run_part1()
        run_part2()
        run_part3()
        run_part4()
    
    print("\n" + "="*70)
    print("ASSIGNMENT COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - models/transformer_classifier.pth")
    print("  - models/abstractive_summarizer.pth")
    print("  - datasets/extractive_summaries.csv")
    print("  - rag_results.json")
    print("\nPlease see the report for detailed analysis and discussion.")


if __name__ == "__main__":
    main()