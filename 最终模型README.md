# Tacotron-2-NVIDIA

## 要求

确保具有以下组件：

NVIDIA Docker（https://github.com/NVIDIA/nvidia-docker）
PyTorch 19.06-py3 + NGC容器或更高版本（https://ngc.nvidia.com/catalog/containers/nvidia:pytorch）
NVIDIA Volta（https://www.nvidia.com/zh-cn/data-center/volta-gpu-architecture/）或Turing（https://www.nvidia.com/zh-cn/geforce/turing/）架构GPU

## 向导
请使用Biao-Bei数据集上Tacrotron 2和WaveGlow模型的默认参数执行以下步骤,以使用具有Tensor Core的混合精度或FP32来训练模型，

1.下载Biao-Bei标准语音数据集BZNYP并进行预处理。 使用./scripts/Biao-Bei_prepare_dataset.sh下载脚本可以自动下载和预处理训练，验证和测试数据集。

    bash scripts/Biao-Bei_prepare_dataset.sh

数据下载到./Biao-Bei目录并安装到NGC容器中的/workspace/tacotron2/Biao-Bei。

2.建造Tacotron 2和WaveGlow PyTorch NGC容器。

    bash scripts/docker/build.sh

3.在NGC容器中启动一个交互式会话以运行训练/推理。 构建容器映像之后，可以使用以下命令启动交互式CLI会话：

    bash scripts/docker/interactive.sh

Interactive.sh脚本要求指定数据集的位置。 要预处理Tacotron 2训练的数据集，请使用./scripts/Biao-Bei_prepare_mels.sh脚本:

    bash scripts/Biao-Bei_prepare_mels.sh

预处理的梅尔谱图存储在./Biao-Bei/mels目录中。

4.开始训练。 要开始训练Tacotron 2，请运行：

    bash scripts/Biao-Bei_train_tacotron2.sh

要开始训练WaveGlow，请运行：

    bash scripts/Biao-Bei_train_waveglow.sh

5.开始验证/评估。 确保您的损失值与“结果”部分表中列出的损失值相当。 对于这两种模型，损耗值都存储在./Biao-Bei_output/nvlog.json日志文件中。

训练Tacotron 2和WaveGlow模型后，应该获得与./audio文件夹中的样本相似的音频结果。 有关生成音频的详细信息，请参见下面的推理过程部分。

训练脚本在每个训练周期后自动运行验证。 验证的结果将打印到标准输出（stdout）并保存到日志文件。

6.开始合成。Tacotron 2和WaveGlow模型训练完成后，可以使用--tacotron2和--waveglow参数传递的各个检查点来执行推理。 Tacotron2和WaveGlow检查点也可以从NGC下载：

https://ngc.nvidia.com/catalog/models/nvidia:tacotron2pyt_fp16/files?version=3

https://ngc.nvidia.com/catalog/models/nvidia:waveglow256pyt_fp16/files?version=2

7.进行合成：

    python inference.py --tacotron2 <Tacotron2_checkpoint> --waveglow <WaveGlow_checkpoint> --wn-channels 256 -o output/ -i phrase/phrase.txt --fp16

语音是通过-i参数传递的文件中的文本行生成的。行数确定合成大小。若要以混合精度运行推理，请使用--fp16标志。输出的音频将存储在-o参数指定的路径中。

## 关于
代码标志: Biao-Bei

提交了跑通标贝数据集的代码, 能够快速合成中文, 但是停顿不自然.

代码标志: PhonePrssCrystal

提交了跑通标贝数据集, 文本处理是经过Crystal过的代码, 能够快速合成中文, 停顿不自然, 但是
使用汉字作为输入, 音调和多音字由于Crystal的问题不自然.


