@echo off
chcp 65001 >nul
echo ============================================================
echo HandTalk AI - 手语识别模型训练
echo ============================================================
echo.
echo 本脚本将:
echo 1. 下载 CSL-News 数据集
echo 2. 采样 5000 条数据
echo 3. 使用 ResNet18 进行迁移学习训练
echo 4. 保存模型到 models/sign_language/
echo.
echo 训练预计需要 10-30 分钟（取决于 GPU）
echo ============================================================
echo.

cd /d "%~dp0.."

REM 激活 conda 环境
echo [1/4] 激活 Conda 环境...
call conda activate Shen
if errorlevel 1 (
    echo 错误: 无法激活 Conda 环境 Shen
    echo 请确保已创建该环境: conda create -n Shen python=3.11
    pause
    exit /b 1
)
echo ✓ Conda 环境已激活
echo.

REM 检查依赖
echo [2/4] 检查依赖...
pip show datasets >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖...
    pip install datasets torch torchvision tqdm scikit-learn
)
pip show torch >nul 2>&1
if errorlevel 1 (
    echo 错误: PyTorch 未安装
    echo 请运行: pip install torch torchvision
    pause
    exit /b 1
)
echo ✓ 依赖检查完成
echo.

REM 运行训练
echo [3/4] 开始训练模型...
echo.
python scripts/train_model.py
if errorlevel 1 (
    echo.
    echo 错误: 训练失败
    pause
    exit /b 1
)
echo.

REM 验证模型
echo [4/4] 验证模型...
if exist "models/sign_language/sign_language_model.pth" (
    echo ✓ 模型训练完成！
    echo ✓ 模型文件: models/sign_language/sign_language_model.pth
) else (
    echo ⚠ 模型文件未找到
)
echo.

echo ============================================================
echo 训练完成！
echo ============================================================
echo.
echo 下一步:
echo 1. 启动后端服务: python -m uvicorn app.main:app --reload
echo 2. 测试识别接口: curl http://localhost:8000/api/v1/recognize/status
echo.
pause

