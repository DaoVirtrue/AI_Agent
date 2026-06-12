# Pitfall 06: 知识图谱维护开销过大 (KG Maintenance Overhead)

## 症状 (Symptoms)

- 知识图谱构建时间随文档量指数增长
- Neo4j/图数据库维护成本远超预期
- 实体和关系提取的LLM API费用爆炸
- 图谱中的关系存在大量噪声（提取错误的关系类型）
- 图遍历检索延迟远超向量检索（2秒 vs 50ms）
- 图谱更新跟不上文档更新频率
- 团队成员不愿维护图谱（太复杂）

## 根本原因 (Root Cause)

1. **过度工程化**: 一开始就构建完整的Neo4j集群和复杂的关系提取
2. **关系提取质量低**: 基于LLM的关系提取准确率 < 70%，产生大量噪声
3. **图规模失控**: 每个文档提取所有可能实体和关系，导致图过于密集
4. **混合检索比例失衡**: 图遍历耗时是向量检索的40倍，却只贡献10%的召回提升
5. **维护成本被低估**: 实体去重、关系冲突解决、图谱一致性检查需要持续的工程投入

## 真实场景 (Real Scenario)

某初创公司决定为其RAG系统构建知识图谱：
- 初期计划：2周构建，1人维护
- 实际情况：
  - 实体提取准确率60%（大量误提取）
  - 关系提取准确率仅45%
  - Neo4j许可证费用每月$5000
  - 图谱更新需要额外8小时/周
  - 图遍历检索延迟2-5秒，远高于200ms目标
  - 3个月后项目被放弃

正确做法：从小规模NetworkX内存图开始，仅在明确ROI后才考虑Neo4j。

## 完整解决方案 (Complete Solution)

```python
"""
知识图谱分级实施策略
Start Small, Scale Gradually
"""

class KGImplementationPhase(Enum):
    """KG实施阶段"""
    PHASE_0 = "no_kg"                # 无KG
    PHASE_1 = "entity_linking"       # 仅实体链接
    PHASE_2 = "simple_relations"     # 简单关系（规则驱动）
    PHASE_3 = "llm_relations"        # LLM驱动关系提取
    PHASE_4 = "full_neo4j"           # 完整Neo4j部署


class GradualKGStrategy:
    """渐进式知识图谱策略"""
    
    def __init__(self):
        self.current_phase = KGImplementationPhase.PHASE_0
        self.gate_criteria = self._define_gates()
    
    def _define_gates(self) -> dict:
        """定义每个阶段的准入标准"""
        return {
            KGImplementationPhase.PHASE_0: {
                "description": "无KG - 纯向量检索基线",
                "metrics_to_collect": ["recall@10", "latency_p99", "user_satisfaction"],
            },
            KGImplementationPhase.PHASE_1: {
                "description": "实体链接 - 仅识别和链接实体",
                "prerequisite": "PHASE_0 基线指标收集 > 2周",
                "success_criteria": "实体链接准确率 > 85%",
                "max_complexity": "NetworkX内存图，实体数 < 10000",
            },
            KGImplementationPhase.PHASE_2: {
                "description": "简单关系 - 基于规则的关系提取",
                "prerequisite": "PHASE_1 实体链接准确率 > 85%",
                "success_criteria": "关系准确率 > 80%，且图遍历延迟 < 100ms",
                "max_complexity": "关系类型 < 5种，关系数 < 50000",
                "cost_per_document": "< $0.001 (仅规则匹配，无LLM)",
            },
            KGImplementationPhase.PHASE_3: {
                "description": "LLM关系 - 使用LLM提取关系",
                "prerequisite": "PHASE_2 运行 > 1个月且检索指标提升 > 5%",
                "success_criteria": "LLM关系精确率 > 90%，且有明确ROI",
                "cost_limit": "LLM API费用 < 总API费用的10%",
            },
            KGImplementationPhase.PHASE_4: {
                "description": "完整Neo4j - 生产级图数据库",
                "prerequisite": "PHASE_3 数据量达到内存上限",
                "success_criteria": "图查询延迟 < 200ms (P99)",
                "infra_cost_limit": "Neo4j费用 < 总基础设施费用的20%",
            },
        }
    
    def evaluate_phase_upgrade(self) -> tuple[bool, str]:
        """评估是否应升级到下一阶段"""
        gate = self.gate_criteria[self.current_phase]
        
        # 检查是否满足当前阶段的成功标准
        metrics = self._collect_metrics()
        
        if not self._meets_criteria(metrics, gate.get("success_criteria", "")):
            return False, "未满足当前阶段成功标准，继续优化"
        
        # 检查是否有明确的ROI
        if self.current_phase != KGImplementationPhase.PHASE_0:
            improvement = self._measure_improvement_from_kg()
            if improvement < 0.05:  # < 5% 提升
                return False, f"KG仅带来 {improvement:.1%} 提升，ROI不足"
        
        return True, f"可以升级到下一阶段"


class PragmaticKGConstructor:
    """务实的KG构建器 - 从小处着手"""
    
    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.relations: list = []
        self.max_entities = 5000  # 从小规模开始
        self.max_relations = 10000
    
    def extract_entities_pragmatic(self, text: str) -> list[dict]:
        """
        务实的实体提取 - 只提取高置信度实体
        原则：宁可漏提取，不可误提取
        """
        entities = []
        
        # 方法1: 基于已知实体列表匹配（100%精确）
        for known_entity in self._get_known_entities():
            if known_entity["name"].lower() in text.lower():
                entities.append({
                    "name": known_entity["name"],
                    "type": known_entity["type"],
                    "confidence": 1.0,
                    "method": "exact_match",
                })
        
        # 方法2: 基于正则规则（高精确）
        rule_entities = self._rule_based_extract(text)
        for re_entity in rule_entities:
            if re_entity not in [e["name"] for e in entities]:
                entities.append({
                    "name": re_entity,
                    "type": "CONCEPT",
                    "confidence": 0.9,
                    "method": "rule",
                })
        
        # 方法3: LLM提取（仅在必要时）
        if self._should_use_llm_extraction(text):
            llm_entities = self._llm_extract_entities(text)
            for le in llm_entities:
                if le["confidence"] > 0.85:  # 高置信度阈值
                    entities.append(le)
        
        # 限制数量
        return entities[:20]  # 每个文档最多20个实体
    
    def extract_relations_pragmatic(self, text: str, entities: list) -> list[dict]:
        """
        务实的关系提取 - 优先规则，LLM辅助
        """
        relations = []
        
        # 方法1: 规则驱动的关系（快速且精确）
        rule_relations = self._rule_based_relations(text, entities)
        relations.extend(rule_relations)
        
        # 方法2: 共现关系（简单高效）
        cooccurrence = self._cooccurrence_relations(entities)
        relations.extend(cooccurrence)
        
        # 方法3: LLM驱动（仅对重要文档）
        if self._is_important_document(text):
            llm_relations = self._llm_extract_relations(text, entities)
            relations.extend(llm_relations)
        
        # 限制数量 + 去重
        return self._deduplicate_relations(relations)[:15]
    
    def _should_use_llm_extraction(self, text: str) -> bool:
        """是否应该使用LLM提取"""
        # 仅在以下情况使用LLM：
        # 1. 文档长度适中（不太短也不太长）
        # 2. 规则提取结果少于3个实体（说明规则覆盖不足）
        # 3. 当前图谱实体数 < 最大实体数的50%
        return (
            200 < len(text) < 5000 and
            len(self.entities) < self.max_entities * 0.5
        )
    
    def _is_important_document(self, text: str) -> bool:
        """文档是否重要（值得LLM提取关系）"""
        importance_signals = [
            "架构" in text or "architecture" in text.lower(),
            "设计" in text or "design" in text.lower(),
            "原理" in text or "principle" in text.lower(),
            len(text) > 1000,
        ]
        return sum(importance_signals) >= 2


def kg_roi_calculator():
    """KG ROI 计算器"""
    
    def calculate_roi(kg_cost_monthly: float, 
                      improvement_metrics: dict,
                      base_traffic: int) -> dict:
        """
        计算知识图谱的投资回报率
        
        Args:
            kg_cost_monthly: KG每月成本 ($)
            improvement_metrics: {"recall_improvement": 0.03, "user_satisfaction_improvement": 0.02, ...}
            base_traffic: 月均查询量
        """
        # 估算收益
        # recall提升3% → 减少用户重新查询3% → 节省API成本
        recall_improvement = improvement_metrics.get("recall_improvement", 0)
        rerun_reduction = base_traffic * recall_improvement * 0.3  # 30%用户会重试
        
        # 用户满意度提升 → 减少流失
        satisfaction_improvement = improvement_metrics.get("user_satisfaction_improvement", 0)
        
        # 简化计算
        api_cost_per_query = 0.002  # $0.002/query
        monthly_savings = rerun_reduction * api_cost_per_query
        monthly_net = monthly_savings - kg_cost_monthly
        
        return {
            "kg_monthly_cost": kg_cost_monthly,
            "estimated_monthly_savings": monthly_savings,
            "net_monthly": monthly_net,
            "roi_pct": (monthly_net / kg_cost_monthly * 100) if kg_cost_monthly > 0 else 0,
            "recommendation": "继续" if monthly_net > 0 else "审查/缩减KG投入",
        }
```

## 检查清单 (Checklist)

- [ ] 从NetworkX内存图开始（而非直接Neo4j）
- [ ] 优先使用基于规则的实体和关系提取
- [ ] LLM提取仅用于高价值文档（长度适中、领域重要）
- [ ] 限制实体和关系数量上限
- [ ] 设置KG准入标准（准确率 > 85% 才入库）
- [ ] 定期计算KG ROI（每月成本 vs 检索指标提升）
- [ ] 图遍历延迟监控（P99 < 200ms）
- [ ] 实体去重和融合（相同实体不同名称）
- [ ] 关系冲突检测和解决
- [ ] 考虑"无KG"响应时间作为基线对比
- [ ] 设置KG的明确退出标准（ROI不足时停用）
