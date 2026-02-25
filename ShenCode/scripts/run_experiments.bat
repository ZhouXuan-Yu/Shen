@echo off
REM ============================================================
REM 自动化实验流程脚本
REM 运行：双击此文件 或 在命令行执行 scripts\run_experiments.bat
REM ============================================================

echo ============================================================
echo 少样本手语识别实验 - 自动化流程
echo ============================================================

REM 激活 conda 环境
call conda activate sjt

REM 设置项目根目录
set PROJECT_ROOT=%~dp0..
cd /d %PROJECT_ROOT%

REM ============================================================
REM 阶段 1: 数据准备（如果尚未完成）
REM ============================================================
echo.
echo [阶段 1] 检查数据准备...
if not exist "data\processed\manifest.jsonl" (
    echo 正在生成 manifest...
    python tools/prepare_dataset.py --autsl
) else (
    echo manifest 已存在，跳过。
)

if not exist "data\splits\kshot_K5_seed0.json" (
    echo 正在生成 K-shot 索引...
    python tools/make_fewshot_splits.py --k 1 --seed 0
    python tools/make_fewshot_splits.py --k 1 --seed 1
    python tools/make_fewshot_splits.py --k 1 --seed 2
    python tools/make_fewshot_splits.py --k 5 --seed 0
    python tools/make_fewshot_splits.py --k 5 --seed 1
    python tools/make_fewshot_splits.py --k 5 --seed 2
) else (
    echo K-shot 索引已存在，跳过。
)

REM ============================================================
REM 阶段 2: P0 训练矩阵（3 模型 × 2K × 3 seeds = 18 次）
REM ============================================================
echo.
echo [阶段 2] 开始 P0 训练矩阵...
echo 共计: 3 模型 x 2 K-shot x 3 seeds = 18 次训练

set MODELS=resnet18 resnet34 mobilenetv2
set K_VALUES=1 5
set SEEDS=0 1 2

for %%m in (%MODELS%) do (
    for %%k in (%K_VALUES%) do (
        for %%s in (%SEEDS%) do (
            echo.
            echo ----------------------------------------
            echo 训练: %%m, K=%%k, seed=%%s
            echo ----------------------------------------
            python tools/train.py --model %%m --k %%k --seed %%s

            REM 训练后立即评估
            set RUN_PATTERN=%%m_K%%k_seed%%s
            for /d %%d in (runs\%%m_K%%k_seed%%s_*) do (
                if exist "%%d\checkpoints\best.pt" (
                    echo 评估: %%d
                    python tools/eval.py --checkpoint "%%d\checkpoints\best.pt" --split val --seed %%s
                )
            )
        )
    )
)

REM ============================================================
REM 阶段 3: 资源评测
REM ============================================================
echo.
echo [阶段 3] 资源评测...
python tools/benchmark.py

REM ============================================================
REM 阶段 4: 汇总结果
REM ============================================================
echo.
echo [阶段 4] 汇总结果...
python tools/summarize_results.py

echo.
echo ============================================================
echo 实验完成！
echo 结果保存在: results\tables\table_main.csv
echo ============================================================
pause
