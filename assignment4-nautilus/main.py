"""
Assignment 4: Transformer and LLMs
Main Entry Point for Nautilus HyperCluster

This script runs all four parts of the assignment:
1. Transformer Classifier for Sentiment Analysis
2. Text Summarization (Extractive + Abstractive)
3. Parameter-Efficient Fine-tuning of TinyLlama
4. Retrieval-Augmented Generation (RAG)

Usage:
    python main.py              # Run all parts
    python main.py --part 1     # Run only Part 1
    python main.py --part 2     # Run only Part 2
    python main.py --part 3     # Run only Part 3 (fine-tuning)
    python main.py --part 4     # Run only Part 4 (RAG)
    python main.py --part 3 4   # Run Parts 3 and 4 only
"""

import os
import sys
import argparse
import traceback


def print_header():
    """Print assignment header."""
    print("=" * 70)
    print("ASSIGNMENT 4: TRANSFORMER AND LLMs")
    print("=" * 70)
    print("""
This assignment covers:
  1. Sequence Classification using Transformer Encoder
  2. Extractive and Abstractive Text Summarization
  3. Parameter-Efficient Fine-tuning of Pretrained LLM (TinyLlama)
  4. Retrieval-Augmented Generation (RAG)
""")


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("PyTorch not installed!")
    print()


def run_part1():
    """Run Part 1: Transformer Classifier."""
    print("\n" + "=" * 70)
    print("PART 1: TRANSFORMER CLASSIFIER FOR SENTIMENT ANALYSIS")
    print("=" * 70)
    
    try:
        # Check if we have the dataset
        dataset_path = "datasets/vectorized_news_skipgram_embeddings.csv"
        if not os.path.exists(dataset_path):
            print(f"\nDataset not found at {dataset_path}")
            print("Skipping Part 1 - please ensure datasets are available")
            return False
        
        from transformer_classifier import main as transformer_main
        sys.argv = ['transformer_classifier.py']
        transformer_main()
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Running transformer_classifier.py directly...")
        os.system("python transformer_classifier.py")
        return True
    except Exception as e:
        print(f"Error in Part 1: {e}")
        traceback.print_exc()
        return False


def run_part2():
    """Run Part 2: Text Summarization."""
    print("\n" + "=" * 70)
    print("PART 2: TEXT SUMMARIZATION")
    print("=" * 70)
    
    success = True
    
    # Part 2.1: Extractive
    print("\n" + "-" * 50)
    print("Part 2.1: Extractive Summarization")
    print("-" * 50)
    
    try:
        from extractive_summary import main as extractive_main
        sys.argv = ['extractive_summary.py']
        extractive_main()
    except ImportError:
        print("Running extractive_summary.py directly...")
        os.system("python extractive_summary.py")
    except Exception as e:
        print(f"Error in Extractive Summarization: {e}")
        traceback.print_exc()
        success = False
    
    # Part 2.2: Abstractive
    print("\n" + "-" * 50)
    print("Part 2.2: Abstractive Summarization")
    print("-" * 50)
    
    try:
        from abstractive_summary import main as abstractive_main
        sys.argv = ['abstractive_summary.py']
        abstractive_main()
    except ImportError:
        print("Running abstractive_summary.py directly...")
        os.system("python abstractive_summary.py")
    except Exception as e:
        print(f"Error in Abstractive Summarization: {e}")
        traceback.print_exc()
        success = False
    
    return success


def run_part3(methods=None, epochs=3, batch_size=4):
    """Run Part 3: Parameter-Efficient Fine-tuning."""
    print("\n" + "=" * 70)
    print("PART 3: PARAMETER-EFFICIENT FINE-TUNING OF TinyLlama")
    print("=" * 70)
    
    if methods is None:
        methods = ["lora", "adapter", "prefix", "full"]
    
    success = True
    
    try:
        import params_finetune
        
        # Train each method
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Training with {method.upper()}")
            print('='*60)
            
            try:
                # Adjust batch size for full fine-tuning
                bs = 1 if method == "full" else batch_size
                ep = 1 if method == "full" else epochs
                
                sys.argv = [
                    'params_finetune.py',
                    '--method', method,
                    '--train',
                    '--batch_size', str(bs),
                    '--epochs', str(ep)
                ]
                params_finetune.main()
                
            except Exception as e:
                print(f"Error training {method}: {e}")
                if method == "full":
                    print("Full fine-tuning failed (expected on limited VRAM)")
                else:
                    traceback.print_exc()
                    success = False
        
        # Evaluate all models
        print(f"\n{'='*60}")
        print("Evaluating all fine-tuned models")
        print('='*60)
        
        sys.argv = ['params_finetune.py', '--evaluate']
        params_finetune.main()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Running params_finetune.py directly...")
        
        for method in methods:
            bs = 1 if method == "full" else batch_size
            ep = 1 if method == "full" else epochs
            cmd = f"python params_finetune.py --method {method} --train --batch_size {bs} --epochs {ep}"
            print(f"\nRunning: {cmd}")
            os.system(cmd)
        
        os.system("python params_finetune.py --evaluate")
        
    except Exception as e:
        print(f"Error in Part 3: {e}")
        traceback.print_exc()
        success = False
    
    return success


def run_part4():
    """Run Part 4: RAG."""
    print("\n" + "=" * 70)
    print("PART 4: RETRIEVAL-AUGMENTED GENERATION (RAG)")
    print("=" * 70)
    
    try:
        import rag_tune
        
        # Build index
        print("\n" + "-" * 50)
        print("Building RAG Index")
        print("-" * 50)
        sys.argv = ['rag_tune.py', '--build']
        rag_tune.main()
        
        # Evaluate
        print("\n" + "-" * 50)
        print("Evaluating RAG System")
        print("-" * 50)
        sys.argv = ['rag_tune.py', '--evaluate']
        rag_tune.main()
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Running rag_tune.py directly...")
        os.system("python rag_tune.py --build")
        os.system("python rag_tune.py --evaluate")
        return True
        
    except Exception as e:
        print(f"Error in Part 4: {e}")
        traceback.print_exc()
        return False


def copy_results_to_data():
    """Copy results to /data for Nautilus PVC."""
    print("\n" + "=" * 70)
    print("COPYING RESULTS TO /data")
    print("=" * 70)
    
    data_dir = "/data"
    if not os.path.exists(data_dir):
        print("/data directory not found (not running on Nautilus?)")
        print("Skipping copy step")
        return
    
    import shutil
    
    # Copy models
    if os.path.exists("models"):
        shutil.copytree("models", f"{data_dir}/models", dirs_exist_ok=True)
        print("Copied models/ to /data/models/")
    
    # Copy results
    for filename in ["finetuning_results.json", "rag_results.json"]:
        if os.path.exists(filename):
            shutil.copy(filename, f"{data_dir}/{filename}")
            print(f"Copied {filename} to /data/")
    
    # Copy datasets
    if os.path.exists("datasets"):
        for f in os.listdir("datasets"):
            if f.endswith(".csv"):
                shutil.copy(f"datasets/{f}", f"{data_dir}/{f}")
                print(f"Copied datasets/{f} to /data/")
    
    print("\nResults in /data:")
    os.system("ls -la /data/")


def print_summary():
    """Print final summary."""
    print("\n" + "=" * 70)
    print("ASSIGNMENT COMPLETE")
    print("=" * 70)
    print("""
Generated files:
  - models/transformer_classifier.pth (Part 1)
  - models/abstractive_summarizer.pth (Part 2)
  - datasets/extractive_summaries.csv (Part 2)
  - models/finetuned/lora/ (Part 3)
  - models/finetuned/adapter/ (Part 3)
  - models/finetuned/prefix/ (Part 3)
  - models/finetuned/full/ (Part 3, if successful)
  - finetuning_results.json (Part 3)
  - models/rag/ (Part 4)
  - rag_results.json (Part 4)

Please see the report for detailed analysis and discussion.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Assignment 4: Transformer and LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run all parts
  python main.py --part 1     # Run only Part 1
  python main.py --part 3 4   # Run Parts 3 and 4
  python main.py --part 3 --methods lora adapter  # Fine-tune with specific methods
        """
    )
    parser.add_argument(
        '--part', 
        nargs='+', 
        type=int, 
        choices=[1, 2, 3, 4],
        help='Which part(s) to run (1, 2, 3, or 4)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['lora', 'adapter', 'prefix', 'full'],
        help='Fine-tuning methods for Part 3'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs for Part 3'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for Part 3'
    )
    parser.add_argument(
        '--skip_copy',
        action='store_true',
        help='Skip copying results to /data'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    check_gpu()
    
    # Determine which parts to run
    parts_to_run = args.part if args.part else [1, 2, 3, 4]
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/finetuned", exist_ok=True)
    os.makedirs("models/rag", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    
    # Run requested parts
    results = {}
    
    if 1 in parts_to_run:
        results[1] = run_part1()
    
    if 2 in parts_to_run:
        results[2] = run_part2()
    
    if 3 in parts_to_run:
        results[3] = run_part3(
            methods=args.methods,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    if 4 in parts_to_run:
        results[4] = run_part4()
    
    # Copy results to /data if on Nautilus
    if not args.skip_copy:
        copy_results_to_data()
    
    # Print summary
    print_summary()
    
    # Print status
    print("\nExecution Status:")
    for part, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  Part {part}: {status}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())