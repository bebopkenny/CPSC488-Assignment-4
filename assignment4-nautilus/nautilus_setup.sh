#!/bin/bash
# setup_nautilus.sh
# Script to set up Assignment 4 for Nautilus deployment
# 
# Usage:
#   chmod +x setup_nautilus.sh
#   ./setup_nautilus.sh

echo "=============================================="
echo "Assignment 4 - Nautilus Setup Script"
echo "=============================================="

# Check if datasets exist
echo ""
echo "Step 1: Checking for dataset files..."

if [ ! -f "datasets/vectorized_news_skipgram_embeddings.csv" ]; then
    echo "WARNING: datasets/vectorized_news_skipgram_embeddings.csv not found!"
    echo "Please copy your dataset files to the datasets/ folder:"
    echo "  cp /path/to/vectorized_news_skipgram_embeddings.csv datasets/"
    echo "  cp /path/to/aggregated_news.csv datasets/"
else
    echo "✓ Dataset files found"
fi

# Build Docker image
echo ""
echo "Step 2: Building Docker image..."
echo "Running: docker build -t bebopkenny/cpsc488-a4:gpu ."

docker build -t bebopkenny/cpsc488-a4:gpu .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully"
else
    echo "✗ Docker build failed!"
    exit 1
fi

# Push Docker image
echo ""
echo "Step 3: Pushing Docker image to Docker Hub..."
echo "Running: docker push bebopkenny/cpsc488-a4:gpu"

docker push bebopkenny/cpsc488-a4:gpu

if [ $? -eq 0 ]; then
    echo "✓ Docker image pushed successfully"
else
    echo "✗ Docker push failed! Make sure you're logged in:"
    echo "  docker login"
    exit 1
fi

# Instructions for Nautilus
echo ""
echo "=============================================="
echo "Setup Complete! Next Steps for Nautilus:"
echo "=============================================="
echo ""
echo "1. Connect to Nautilus cluster"
echo ""
echo "2. (First time only) Create PVC if you don't have one:"
echo "   kubectl apply -f k8s-pvc.yaml"
echo ""
echo "3. Submit the training job:"
echo "   kubectl apply -f k8s-finetune-job.yaml"
echo ""
echo "4. Monitor the job:"
echo "   kubectl get jobs -n csuf-titans"
echo "   kubectl logs -f job/cpsc488-finetune -n csuf-titans"
echo ""
echo "5. After completion, retrieve results:"
echo "   kubectl apply -f k8s-pvc-access.yaml"
echo "   kubectl cp csuf-titans/pvc-reader:/data/finetuning_results.json ./results/finetuning_results.json"
echo "   kubectl cp csuf-titans/pvc-reader:/data/rag_results.json ./results/rag_results.json"
echo ""
echo "=============================================="