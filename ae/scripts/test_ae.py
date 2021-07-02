import sys
from ae.pipeline.simple_ae_pipeline import SimpleAEPipeline

def main():
    cmd = "main --config /home/swei20/AE/configs/ae/train/test_config.json"
    sys.argv = cmd.split()
    print(sys.argv)

    p=SimpleAEPipeline()
    p.execute()

if __name__ == "__main__":
    main()
