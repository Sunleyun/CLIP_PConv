import subprocess
import sys

def run_train():
    # 将命令参数封装在列表中
    # 使用 r"" 原始字符串可以避免 Windows 路径反斜杠的转义问题
    cmd = [
        "python", "train.py",
        "--train_lmdb", "/data/shiQ_train.lmdb",
        "--val_lmdb", "/data/shiQ_test.lmdb",
        "--stage", "stage3",
        "--batch_size", "8",
        "--num_workers", "8",
        "--epochs", "1",
        "--lr", "2e-4",
        "--logdir", "runs/shiq_stage3",
        "--outdir", "outputs/shiq_stage3"
    ]

    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # check=True 表示如果命令运行失败会抛出异常
        # shell=False 在 Windows 下推荐使用列表传参，更安全
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练脚本运行出错: {e}")
    except KeyboardInterrupt:
        print("\n用户停止训练。")

if __name__ == "__main__":
    run_train()
