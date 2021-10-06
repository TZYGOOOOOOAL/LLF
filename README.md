# 简单实现 LLF （Local Laplacian Filters） 以及 Fast LLF

## 注意插值过程
y = (1-a) * x_i + a * x_{i+1}  
可记为：  
```
y = 0;  
for i in range(N) 
	y += a_i * x_i
```
其中 $a_i$ = |x - x_i| / step  (a为[0,1]插值加权系数)

## remap 函数
注意fast LLF原MATLAB实现 remap函数 与 原始LLF不同，
两者都是关于原图像素 i 的函数，不要被g0迷惑。

fast llF 的映射函数计算之所以不加 i 主要为
Laplace(x+y) = Laplace(x) + Laplace(y)
而Laplace(x)即为原图的Laplace 金字塔

## TODO:
### 1. Remap 函数的参数传入
### 2. 三通道数据处理
此处是采用三个通道单独处理，原LLF不是，需查看原文
### 3. 过程化简
### 4. 金字塔创建 数据类型