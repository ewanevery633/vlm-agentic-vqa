import argparse
import logging
from src.utils import setup_logging
from src.models import QwenVLM
from src.data_loader import load_dataset, show_data
from src.zero_shot import run_zero_shot
from src.agent_pipelines.classic_agent import run_classic_agent_pipeline
from src.agent_pipelines.dl_agent import run_dl_agent_pipeline

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="VLM Agent Evaluation Script")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['zero_shot', 'classic', 'dl', 'all', 'show_sample'],
        help="Evaluation mode to run."
    )
    parser.add_argument(
        '--data_csv',
        type=str,
        default='data/vqa_dataset.csv',
        help="Path to the VQA dataset CSV file."
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default='data/images',
        help="Path to the directory containing images."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        help="Name of the Hugging Face model to use."
    )
    parser.add_argument(
        '--sample_index',
        type=int,
        default=0,
        help="Index of the sample to show if mode is 'show_sample'."
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting script in mode: {args.mode}")

    # --- 1. Load Dataset ---
    dataset = load_dataset(args.data_csv, args.images_dir)
    if dataset is None:
        logger.error("Failed to load dataset. Exiting.")
        return

    # --- 2. Handle 'show_sample' mode ---
    if args.mode == 'show_sample':
        show_data(args.sample_index, dataset)
        return

    # --- 3. Load Model(s) ---
    # We load the main VLM and a separate instance for the judge
    # to avoid context pollution, although they can be the same model.
    try:
        logger.info("Loading main VLM...")
        vlm = QwenVLM(args.model_name)
        logger.info("Loading Judge VLM...")
        judge_vlm = QwenVLM(args.model_name)
    except Exception as e:
        logger.error(f"Failed to load VLM models: {e}. Exiting.")
        return

    # --- 4. Run Evaluations ---
    results = {}

    if args.mode in ['zero_shot', 'all']:
        try:
            acc, _ = run_zero_shot(vlm, dataset, judge_vlm)
            results['Zero-Shot'] = acc
        except Exception as e:
            logger.error(f"Error during Zero-Shot evaluation: {e}", exc_info=True)

    if args.mode in ['classic', 'all']:
        try:
            acc, _ = run_classic_agent_pipeline(vlm, dataset, judge_vlm)
            results['Classic Agent'] = acc
        except Exception as e:
            logger.error(f"Error during Classic Agent evaluation: {e}", exc_info=True)

    if args.mode in ['dl', 'all']:
        try:
            acc, _ = run_dl_agent_pipeline(vlm, dataset, judge_vlm)
            results['DL Agent'] = acc
        except Exception as e:
            logger.error(f"Error during DL Agent evaluation: {e}", exc_info=True)
            
    # --- 5. Print Final Report ---
    logger.info("--- 📊 Evaluation Report ---")
    print("\n--- 📊 Evaluation Report ---")
    if not results:
        logger.warning("No evaluations were run.")
        print("No evaluations were run.")
    else:
        for mode, accuracy in results.items():
            report_line = f"[{mode}] Accuracy: {accuracy:.2%}"
            logger.info(report_line)
            print(report_line)
    logger.info("--- End of Report ---")

if __name__ == "__main__":
    main()
