# 前端快速启动指南

## 已完成的工作

✅ **完整的前端应用已创建**，包含以下功能：

### 核心功能
1. **仪表盘 (Dashboard)** - 系统概览和统计
2. **回测 (Backtest)** - 策略回测配置和结果展示
3. **历史 (History)** - 回测历史查询和管理
4. **监控 (Monitoring)** - 系统状态监控
5. **设置 (Settings)** - 主题和语言设置

### 技术特性
- ✅ React 18 + TypeScript
- ✅ Vite 构建工具（快速开发）
- ✅ Tailwind CSS（现代化样式）
- ✅ 暗色/亮色主题切换
- ✅ 中英文国际化
- ✅ 响应式设计（支持移动端）
- ✅ Recharts 图表库
- ✅ React Router 路由
- ✅ Zustand 状态管理

## 快速启动

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 配置环境变量（可选）

创建 `frontend/.env` 文件：

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

如果不设置，默认会使用 `/api/v1`（通过Vite代理）

### 3. 启动开发服务器

```bash
npm run dev
```

前端将在 http://localhost:3000 启动

### 4. 确保后端API运行

确保后端API在 http://localhost:8000 运行：

```bash
# 在项目根目录
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 项目结构

```
frontend/
├── src/
│   ├── components/          # 可复用组件
│   │   ├── Layout.tsx      # 布局组件（导航栏等）
│   │   ├── BacktestForm.tsx    # 回测表单
│   │   └── BacktestResults.tsx # 回测结果展示
│   ├── pages/               # 页面组件
│   │   ├── Dashboard.tsx   # 仪表盘
│   │   ├── Backtest.tsx    # 回测页面
│   │   ├── History.tsx     # 历史页面
│   │   ├── Monitoring.tsx  # 监控页面
│   │   └── Settings.tsx    # 设置页面
│   ├── services/           # API服务
│   │   └── api.ts          # Axios封装
│   ├── stores/             # 状态管理
│   │   └── themeStore.ts   # 主题状态
│   ├── i18n/               # 国际化
│   │   └── config.tsx      # i18n配置
│   ├── App.tsx             # 主应用组件
│   └── main.tsx            # 入口文件
├── index.html
├── package.json
├── vite.config.ts          # Vite配置（包含API代理）
└── tailwind.config.js      # Tailwind配置
```

## 功能说明

### 1. 仪表盘
- 显示总回测数、活跃策略数等统计
- 快速操作入口

### 2. 回测页面
- 左侧：回测配置表单
  - 选择策略
  - 设置标的、时间周期
  - 配置日期范围（天数或日期范围）
  - 设置初始资金、手续费、滑点
- 右侧：实时结果展示
  - 性能指标卡片
  - 组合价值曲线图
  - 价格与交易信号图

### 3. 历史页面
- 回测历史列表
- 按策略/标的筛选
- 查看/删除操作

### 4. 监控页面
- 系统健康状态
- 数据质量指标（待完善）
- 性能指标（待完善）

### 5. 设置页面
- 主题切换（亮色/暗色）
- 语言切换（中文/英文）

## 界面特性

### 主题配色
- **亮色模式**：白色背景，深色文字，蓝色主色调
- **暗色模式**：深灰背景，浅色文字，蓝色主色调
- 自动保存用户偏好

### 国际化
- 支持中文和英文
- 所有文本都通过i18n管理
- 语言切换立即生效

### 响应式设计
- 桌面端：完整布局
- 平板：自适应列数
- 移动端：单列布局，折叠导航

## 下一步优化建议

1. **增强图表功能**
   - 添加更多图表类型（K线图、热力图等）
   - 支持图表交互（缩放、筛选）

2. **完善监控页面**
   - 实时数据刷新
   - 告警列表展示
   - 系统性能指标

3. **添加更多功能**
   - 策略管理页面
   - 因子分析页面
   - 数据管理页面

4. **性能优化**
   - 虚拟滚动（长列表）
   - 代码分割
   - 图片懒加载

5. **用户体验**
   - 加载状态优化
   - 错误提示优化
   - 操作确认对话框

## 常见问题

### Q: 前端无法连接后端API？
A: 确保：
1. 后端API在 http://localhost:8000 运行
2. Vite配置中的proxy设置正确
3. 检查浏览器控制台的错误信息

### Q: 样式不生效？
A: 确保：
1. Tailwind CSS已正确安装
2. `index.css` 中导入了Tailwind指令
3. 重启开发服务器

### Q: 国际化不生效？
A: 确保：
1. i18next已正确配置
2. 翻译文件已加载
3. 组件中使用了 `useTranslation` hook

## 构建生产版本

```bash
npm run build
```

构建产物在 `dist/` 目录，可以部署到任何静态文件服务器。

## 总结

✅ **前端应用已完整创建**，包含：
- 5个主要页面
- 完整的路由和导航
- 主题和国际化支持
- API集成
- 响应式设计

现在可以：
1. 运行 `npm install` 安装依赖
2. 运行 `npm run dev` 启动开发服务器
3. 在浏览器访问 http://localhost:3000 查看界面

前端已准备好与后端API集成使用！
