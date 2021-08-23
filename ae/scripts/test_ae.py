import sys
from ae.pipeline.ae_pipeline import AEPipeline

def main():
    # cmd = "main --config /home/swei20/AE/configs/test_config.json"

    cmd = "main --config /home/swei20/AE/configs/ae/hier/pca_config.json"
    sys.argv = cmd.split()
    print(sys.argv)

    p=AEPipeline()
    p.execute()

if __name__ == "__main__":
    main()
