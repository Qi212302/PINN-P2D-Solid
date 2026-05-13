# PINN 求解 P2D 固相扩散方程

本项目实现了 P2D 模型中球形活性颗粒的固相锂扩散方程。假设固相扩散系数 `D_s` 为常数，颗粒半径为 `R_s`。

## 控制方程

球坐标下的固相扩散方程为

```text
∂c_s/∂t = D_s / r^2 * ∂/∂r(r^2 ∂c_s/∂r),     0 < r < R_s
```

等价写成

```text
∂c_s/∂t = D_s (∂²c_s/∂r² + 2/r * ∂c_s/∂r)
```

采用无量纲变量

```text
x = r / R_s
tau = D_s t / R_s²
c = (c_s - c_s0) / c_scale
```

得到 PINN 训练使用的方程

```text
∂c/∂tau = ∂²c/∂x² + 2/x * ∂c/∂x,     0 < x < 1
```

## 初始条件与边界条件

这里采用适合恒流充放电颗粒扩散问题的一组条件：

```text
c(x, 0) = 0
∂c/∂x(0, tau) = 0
∂c/∂x(1, tau) = -phi(tau)
```

其中中心边界 `∂c/∂x(0, tau)=0` 来自球心对称性。表面通量边界对应 P2D 中固液界面反应给颗粒表面的锂通量：

```text
-D_s ∂c_s/∂r |(r=R_s) = J_surf(t)
```

无量纲后为

```text
∂c/∂x(1, tau) = -J_surf R_s / (D_s c_scale) = -phi(tau)
```

当前脚本默认 `phi=0.2`，即恒定脱嵌/嵌入通量的示例。若要接入完整 P2D 模型，可把 `src/pinn_solid_diffusion.py` 中的 `phi_of_tau` 替换为由电流密度或 Butler-Volmer 反应通量给出的时间函数。

## 运行

```bash
python3 src/pinn_solid_diffusion.py
```

可调参数示例：

```bash
python3 src/pinn_solid_diffusion.py --tau-max 0.4 --phi 0.2 --adam-steps 2500 --lbfgs-steps 300
```

输出文件位于 `results/`：

- `solid_diffusion_pinn.pt`：训练后的 PyTorch 权重
- `solid_diffusion_pinn_solution.npz`：PINN 解、有限体积参考解、训练历史
- `concentration_profiles.png`：不同时间的径向浓度分布
- `concentration_field.png`：`x-tau` 平面浓度场
- `metrics.txt`：相对有限体积参考解的误差指标

## 损失函数

当前版本采用硬约束网络：

```text
c_PINN(x, tau) = tau * N(2x² - 1, 2tau/tau_max - 1)
```

因此初始条件 `c(x,0)=0` 和球心对称边界 `∂c/∂x(0,tau)=0` 在网络结构上严格满足。训练损失主要由两部分组成：

```text
L = L_pde + 20 L_flux
```

其中 `L_pde` 是控制方程残差，`L_flux` 是颗粒表面通量边界残差。脚本还内置了有限体积参考解，用于检查 PINN 解的形状和误差。

表面通量损失权重经过敏感性实验比较。实验扫描 `lambda_flux = 1, 5, 10, 20, 40, 80, 100`，完整结果保存在 `results/flux_weight_sensitivity/`。在本算例中，`lambda_flux=20` 不是单一指标的绝对最优，但在参考解 RMSE、表面通量误差、质量守恒误差和 PDE 残差之间给出了更均衡的结果，因此作为默认权重。

## 结果可信性检查

完整输出包含：

- `concentration_profiles.png`：PINN 与有限体积参考解的径向浓度曲线对比
- `error_field.png`：PINN 相对参考解的点误差
- `mass_conservation.png`：体平均浓度与通量积分守恒关系对比
- `pde_residual_field.png`：`log10(|PDE residual|)` 残差场
- `loss_history.png` 与 `loss_history.csv`：训练损失历史
- `metrics.txt`：误差、守恒、边界和残差指标

注意：恒定通量从 `tau=0+` 瞬时施加时，初始均匀浓度与表面通量边界在 `(x=1,tau=0)` 角点不相容。因此 `metrics.txt` 同时给出全域指标，以及排除 `tau < 1e-4` 初始角点后的指标。论文中应明确说明这一点，不应把角点尖峰解释为普通训练误差。
