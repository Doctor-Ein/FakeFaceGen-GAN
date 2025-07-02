### 前排提醒
我知道bouns分支打错了😭 -> bonus ✅

### 前言与实现效果展示
生成对抗网络**GAN**（**Generative Adversarial Networks）**，核心思路是在同一个框架下交替训练一个判别器，一个生成器，让它们相互“对抗”。生成器以生成贴近数据集的图像为目标，判别器以精确判别真正属于数据集的图像为目标，最终达到生成的图片相当逼真的效果。

此次小项目正是在此基础上的**DCGAN**的网络框架的实践，具体效果如下：

**第一次，基础的网络框架：**![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389528674-a79658df-9689-43b5-aad1-793cba875071.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389528689-73e6c515-2b64-4fbc-9df7-aceebd7cac4d.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389528712-d917121b-3a72-4a4a-b2ab-7fd3df9d92be.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389528680-34357960-a5b5-4985-b4b5-32fb1dd3abdd.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389528701-309ef612-9109-4796-8786-fd17c723e752.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389528835-b0308de0-080e-4d10-8bd9-a97953204368.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389528843-619f7dea-796c-46c8-ad8d-ed49a7484e1f.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389528864-66225643-a767-4612-9789-69f2871db9cd.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389568728-e8cf3b76-dd9e-4ec8-89e5-a0403d2dc9f1.png)

**第二次，添加正则化技术：**![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389324675-f04dcc50-b4bb-4aaf-85ec-9143b77e8697.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389324718-ebf414d3-5fed-4f01-a70e-3be20cd90a03.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389339143-3084c795-427f-482d-80dd-19334df44642.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389339172-7dae2f15-aee5-42d0-9d65-cbfa548faad2.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389339178-aa6a324e-885f-495f-904b-c37e37a6ebe7.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389339194-aa9aad3b-7807-4521-8c88-4b1cacf1b631.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389339171-a603f34a-3175-4aa6-abe9-ee15ec4dd938.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373518-4883a5fe-5535-43f1-8c07-20c55e456697.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373513-d8b70d5f-bcc2-446d-848c-bf54cc9f3cc4.png)

**第三次，微调学习率：**![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373513-0958e769-14dc-421a-b204-6285777be28c.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373514-920c2dc8-7165-4a58-93b1-8e1e10ae51ab.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373655-5ba2bcfc-d7dd-446d-8f0a-649f3431816d.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373662-5446b1f7-8b70-4d72-b1a6-aee34eb4f51e.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373682-c9b6b1b1-89c7-4d6e-ba3b-7dc8dd3ccb6c.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373678-d9a5ac6e-fd81-4ec9-98c7-ec211c0696e6.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389373684-5a00bf32-e90a-4ff9-bbe8-85a3a89d6b4e.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389390939-24a00549-b1c6-4302-90ee-2e8fbea85dda.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751389390936-6321379c-f23c-48c1-ac1b-6972b06eb5c2.png)

**第四次，高分辨率：（128 x 128）**

![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751393167651-d3db32a1-a313-4da0-9d3a-04d3eeee4e95.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751393167653-cb83cbd6-f05d-4276-a9bc-8df18d08c9bd.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751393167662-813518c3-3e85-49b6-a612-4f0d0091acc2.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751393167660-f87719b7-04fd-4ecc-8b2a-136231ebc095.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751393167650-3b5096db-e9ec-432e-b237-88a2d817fa03.png)![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751393167820-06785337-766e-48c0-ba7e-a2d4f5697793.png)

P.S. 还是有些特挑的，但基本都是直接选取的连续区域。主要是觉得特别伪人的脸还是吓人的哈，主观感觉正常的人脸大概60-70 %（？）

---

### 实现过程与细节
#### 实现过程
1. 过了遍论文，了解网络结构部分的内容。助教也非常贴心地把网络结构图，以及论文中对于网络搭建的提醒都摘录到**README**中了🥰
2. 通过**jpynb**基本了解了如何定义网络层，怎么调用，参数是什么意义。参考论文的提示和架构图，清楚网络如何搭建，接下来就是动手实现具体的网络结构。
3. 网络的实现其实也并非复杂。（个人见解）主要是关注前后的连接是否正确，主要掌握尺寸，通道数等的计算公式，了解各网络层分别有什么用即可。
4. 在完成的基础工作之上，根据训练结果，又做了一些优化（包括权重预设，超参数设置，防过拟合技术等，后面有详细叙述）
5. 实现更高分辨率的效果（另一个git分支）
6. 整理数据和代码

#### `train.py`的大致思路
最核心的思想就是~~两句话~~三句话：

+ **生成器**试图生成越来越逼真的图像来欺骗判别器。
+ **判别器**试图区分真实图像和生成图像。
+ 两者通过对抗学习共同提升，最终目标是生成器能输出逼真的图像。

接下来按流程讲一讲，首先是初始化预设的部分

```python
netD = Discriminator().to(device)  # 判别器
netG = Generator().to(device)      # 生成器
criterion = nn.BCELoss()           # 二分类交叉熵损失
optimizerD = optim.Adam(...)       # 判别器优化器
optimizerG = optim.Adam(...)       # 生成器优化器
```

这里选定的损失函数是**二分类交叉熵优化**，因为判别器本质是一个二分类器。

单次迭代的具体训练步骤如下：

+ 首先是训练**判别器 D**

```python
optimizerD.zero_grad() # 清空判别器梯度，避免梯度累积
real_cpu = data.to(device) # 将数据迁到至GPU
current_batch_size = real_cpu.size(0) # 获取当前批次大小

# 创建全1标签（真实样本标签）
label = torch.full((current_batch_size,), real_label, dtype=torch.float, device=device)
output = netD(real_cpu).view(-1)
errD_real = criterion(output, label) # 计算真实数据的损失
errD_real.backward() # 反向传播计算梯度（真实数据部分）

# 生成随机噪声作为生成器输入
noise = torch.randn(current_batch_size, G_DIMENSION, 1, 1, device=device)
fake = netG(noise)
label.fill_(fake_label)
output = netD(fake.detach()).view(-1) # detach阻断生成器梯度
errD_fake = criterion(output, label) # 计算假数据的损失
errD_fake.backward() # 反向传播计算梯度（假数据部分）

errD = errD_real + errD_fake # 合并两部分损失
optimizerD.step() # 更新判别器参数
```

+ 接着是训练**生成器 G**

```python
label.fill_(real_label)             # 希望D将生成图像判为真
output = netD(fake_images).view(-1) # 此时不detach ✅
errG = criterion(output, label)
errG.backward()
optimizerG.step()
```

+ 过程中还需要记录损失函数的变化，绘制图像。最后将生成器的参数保存在`generator.params`

### 额外工作（Bonus）
#### 权重初始化
此方面工作来自助教贴出的论文原文

> <font style="color:rgb(31, 35, 40);">All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.</font>
>

观察了一下似乎没有预先实现。于是通过`torch.nn.init`模块实现初始化，如下（在`network.py`中）

```python
import torch.nn.init as init

def weights_init(m):
    """
    初始化权重：从 N(0, 0.02) 的正态分布采样
    适用于：Conv2d, ConvTranspose2d, BatchNorm2d
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init.normal_(m.weight.data, 0.0, 0.02)  # 均值0，标准差0.02
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)  # 偏置初始化为0

# 使用的话，在定义完网络之后即可apply
def __init__(self):
    super().__init__()
    self.main = nn.Sequential(...)
    self.apply(weights_init) # 初始化权重
```

#### 超参数调整
和network相关的超参数是`img_dim`图片维度、`G_DIMENSION`噪声维度，和网络结构的设定有关。在后面更高分辨率的版本中有涉及。

关于学习率`lr`，和`beta1`，`beta2`这些参数一些了解。在观察了第一次的训练结果的记录图（左图）

![图1](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751376367550-06c4b594-d78f-4619-8195-28b3d468896d.png)![图2：减小了判别器的学习率](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751376340370-cabb93c4-4365-4c55-afeb-872c8a93e714.png)

（个人理解）判断问题在于：判断器太强了，因为**D loss**迅速减小到0了，因此减小了其学习率。后续结果表明，这种超参数组合下，生成器和判别器的损失函数**更好的相交错**。并且生成器的收敛程度也比上一轮好得多。

```python
optimizerD = optim.Adam(netD.parameters(), 
                        lr=lr - 0.0001, 
                        betas=(beta1, beta2),
                        weight_decay=1e-4) # 降低判别器的强度，不然生成器很难学习🤔
optimizerG = optim.Adam(netG.parameters(),
                        lr=lr, 
                        betas=(beta1, beta2),
                        weight_decay=1e-6) # 更弱的正则化
```

#### 关于判断生成质量的标准
另外提一下，有一个bonus是要求提出判断生成质量的标准。这个我虽暂时没有什么idea，但是从另一个角度看：论文中作者也提到在空间中存在唯一的最终解（即判断器最后输出结果趋于1/2）

那么过程中类似图2的相互交错，生成器和判断器**真正相互对抗**，才是理论上更好的对抗生成网络的训练过程。**从训练过程反映出训练的质量的观点来看，**可以定义某种指标衡量类似图2的“交错”程度。这个我认为是一个判断生成质量的指标，或者更贴切是训练效果的指标叭🤔

后面查了一下，早期的GAN的判断标准，一个经典的做法是：**<font style="color:rgb(64, 64, 64);">Inception Score (IS) . </font>**使用预训练的Inception网络计算生成样本的类别分布：

+ **质量：** 若生成样本的类别预测概率$ p(y \left| \right. x) $高度集中（低熵），说明样本清晰可辨。
+ **多样性：** 若所有生成样本的类别分布$  p(y) $均匀（高熵），说明样本覆盖多个类别。
+ **汇总公式：** $ \mathrm{IS}=\exp \left(\mathbb{E}_{x \sim P_{g}}[\mathrm{KL}(p(y \mid x) \| p(y))]\right) $

这种从数学上通过分布的指标反映生成质量，显然更为合理Orz.

#### 更高的分辨率
设置目标分辨率为`128 * 128`，主要是因为还是2的倍数，使用如下的配置比较简单，每一次尺寸除以2。

$ \textbf{kernel_size} = 5, \textbf{stride} = 2, \textbf{padding} = 2 \Rightarrow 
\left \lfloor \frac{H - 5 + 2 * 2}{2} \right \rfloor  + 1  = 
\left \lfloor \frac{H + 1}{2} \right \rfloor $

因此网络层数会加深一层，不过注意到最后一层，尺寸仅为4 x 4 小于卷积核了，就一次到位直接卷积到 1x1 激活就OK了。

对于更高的分辨率的情形下，生成的效果较差了，一方面是学习的轮次可能不太够，另一方面则是判别器需要增强，保证生成器能够有较好的训练效果。

### 其他有趣的事
#### 当你挑战高分辨率的，修改网络结构设计新的结构，需要好好计算公式👍
> 本质上来说,`nn.Conv2d`在做如下的事情：
>
> 把输入为`[N, C, H, W]`的张量，经过卷积核的处理后，
>
> 输出为 `[N, output_channels, (H - kernel_size + 2 * padding) // stride + 1, (W - kernel_size + 2 * padding) // stride + 1]`的张量。
>

#### 然后反复验证网络结构没有问题，并且在测试输出形状的确没问题之后，就需要考虑是不是`dataloader.py`没有修改了😭
#### 计算平台的趣事
我本身是 **Macbook Air (M2) **性能只能说是很差啦，硬跑哪怕用上 **mps **也得跑8个小时才有**5 epoch**😆

后面找了一个云平台，用的 **V100-32G，¥1.98/h **不知道学长觉得算不算亏了🤔 跑一个** 64^2 **分辨率的大概就是**5min/epoch **，跑** 128^2 **则是** 7.5 min/epoch. **感觉是正常可接受范畴 ✅

还有试用了 **A100**，明明是性能更高的显卡但时间却和 **V100**一轮差不多乃至更差。因为它的GPU占用率很波动，这一点感觉很奇怪，不知道为何不稳定🤔

![](https://cdn.nlark.com/yuque/0/2025/png/50620204/1751391681814-90090861-1a8b-4f0f-939c-bb61c759fd7f.png)
