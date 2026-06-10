# Search 工具说明

`myagent/search` 已合并为 MyAgent 项目的一部分。

详细功能、安装依赖、启动命令、深挖参数、OCR/图片理解、输出目录说明，请查看：

- `myagent/README.md`

常用启动方式：

```powershell
cd F:\study\研究生\字节竞品分析Agent
$env:PYTHONPATH=(Resolve-Path .).Path
python -m myagent.search.search_cli --query "飞书 竞品分析"
```

输出默认写入：

```text
myagent/search/outputs/<timestamp>/
```
