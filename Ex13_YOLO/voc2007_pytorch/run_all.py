"""
Run training then visualization. Execute from voc2007_pytorch folder:
  python run_all.py
"""
import train
import visualize

if __name__ == "__main__":
    print("=== Training ===")
    train.main()
    print("=== Visualization ===")
    visualize.plot_curves()
    visualize.visualize_predictions(num_samples=8)
    print("Done. Check checkpoints/ and visualizations/")
