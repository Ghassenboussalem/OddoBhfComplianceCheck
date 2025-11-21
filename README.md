# AI-Enhanced Compliance Checker

> Hybrid AI-powered compliance validation system for financial fund documents

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-proprietary-red.svg)]()

---

## 🎯 Overview

The **AI-Enhanced Compliance Checker** is a sophisticated system that validates financial fund documents against regulatory requirements (AMF, ESMA, MAR, SFDR). It combines:

- **AI Semantic Understanding**: Deep context analysis using LLMs
- **Rule-Based Validation**: Fast, reliable pattern matching
- **Hybrid Architecture**: Best of both worlds with confidence scoring

### Key Features

✅ **100% Backward Compatible** - Works with existing workflows  
✅ **Multi-Language Support** - French and English  
✅ **Confidence Scoring** - 0-100 confidence for each finding  
✅ **Intelligent Caching** - Reduces redundant AI calls  
✅ **Automatic Fallback** - Falls back to rules if AI unavailable  
✅ **Comprehensive Checks** - 8 compliance categories, 100+ rules  

---

## 🚀 Quick Start

### Installation

1. **Clone or download the project**

2. **Install dependencies** (if any - check requirements.txt)

3. **Configure API keys** in `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   TOKENFACTORY_API_KEY=your_token_factory_key_here
   ```

### Basic Usage

```bash
# Standard check (rules only)
python check.py exemple.json

# Hybrid AI+Rules mode (recommended)
python check.py exemple.json --hybrid-mode=on

# With custom confidence threshold
python check.py exemple.json --ai-confidence=80

# Show performance metrics
python check.py exemple.json --show-metrics
```

### Output

Results are saved to `exemple_violations.json` with:
- Detailed violation information
- Evidence and location
- Confidence scores (when AI enabled)
- AI reasoning (when AI enabled)
- Categorized by type

---

## 📋 Compliance Checks

### Structure Validation
- Promotional document mention
- Target audience specification
- Fund name presence
- Date validation
- Management company legal mention
- Risk profile information

### Performance Rules
- Performance placement validation
- Benchmark comparison requirements
- Mandatory disclaimers
- Historical vs predictive claims

### Securities/Values (MAR Compliance)
- Investment recommendation detection
- Security mention analysis
- Prohibited language detection
- Context-aware intent analysis

### Prospectus Matching
- Strategy consistency
- Benchmark validation
- Investment objective matching
- Data consistency verification

### Registration Compliance
- Country authorization validation
- Distribution claim detection
- Multi-country support

### General Rules
- Source citations
- Glossary requirements
- Morningstar rating dates
- Technical term validation

### ESG Compliance
- Content distribution analysis
- SFDR classification validation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Layer 1: Rule-Based Pre-filtering     │
│  (Fast keyword/pattern matching)        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Layer 2: AI Semantic Analysis          │
│  (Deep context understanding)           │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Layer 3: Confidence Scoring            │
│  (Combines AI + Rules for final result) │
└─────────────────────────────────────────┘
```

### Core Components

- **check.py** - Main entry point
- **check_hybrid.py** - Hybrid integration layer
- **hybrid_compliance_checker.py** - Main hybrid checker
- **ai_engine.py** - AI service abstraction
- **confidence_scorer.py** - Result combination
- **error_handler.py** - Error handling & fallback
- **config_manager.py** - Configuration management
- **agent.py** - Legacy rule engine

---

## ⚙️ Configuration

### Enhancement Levels

Edit `hybrid_config.json` to set the AI enhancement level:

| Level | Description | Use Case |
|-------|-------------|----------|
| `disabled` | Rules only | Testing, fallback |
| `minimal` | Critical checks only | Conservative start |
| `standard` | Most checks | Recommended |
| `full` | All checks | Maximum accuracy (default) |
| `aggressive` | AI-first | Experimental |

### Quick Configuration

```json
{
  "ai_enabled": true,
  "enhancement_level": "full",
  "confidence": {
    "threshold": 70,
    "high_confidence": 85,
    "review_threshold": 60
  },
  "cache": {
    "enabled": true,
    "max_size": 1000
  }
}
```

### Environment Variables

```bash
export HYBRID_AI_ENABLED=true
export HYBRID_ENHANCEMENT_LEVEL=full
export HYBRID_CONFIDENCE_THRESHOLD=75
```

---

## 📊 Example Output

```json
{
  "document_info": {
    "filename": "exemple.json",
    "fund_name": "ODDO BHF Algo Trend US",
    "processing_mode": "hybrid_ai_rules",
    "ai_enabled": true
  },
  "summary": {
    "total_violations": 38,
    "critical_violations": 22,
    "major_violations": 15,
    "warnings": 1,
    "avg_confidence": 87.5
  },
  "violations_by_category": {
    "STRUCTURE": {
      "count": 3,
      "violations": [
        {
          "rule_id": "STRUCT_003",
          "severity": "CRITICAL",
          "message": "Promotional document mention missing",
          "confidence": 95,
          "ai_reasoning": "No promotional indication found on cover page",
          "status": "VERIFIED_BY_BOTH",
          "location": "Cover Page",
          "evidence": "Add 'Promotional Document' designation"
        }
      ]
    }
  }
}
```

---

## 📚 Documentation

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Complete project report
- **[QUICK_START.md](QUICK_START.md)** - 5-minute setup guide
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - API reference
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Integration instructions
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Configuration options
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migration from legacy
- **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)** - Common issues

---

## 🧹 Project Cleanup

To remove unnecessary files and clean up the project:

```bash
python cleanup_project.py
```

This will remove:
- Redundant/obsolete files
- Example/demo files
- Old test result files
- Redundant documentation
- Cache directories

---

## 🔧 Troubleshooting

### AI Not Working

1. Check API keys in `.env`
2. Test connection: `python -c "from ai_engine import AIEngine; AIEngine().test_connection()"`
3. Check configuration in `hybrid_config.json`

### Low Confidence Scores

- Adjust threshold: Set `confidence.threshold` to 60 in `hybrid_config.json`
- Review manually: Violations with confidence < 70% need review

### Slow Performance

- Enable caching: `cache.enabled: true`
- Use minimal mode: `enhancement_level: "minimal"`
- Reduce batch size

---

## 📈 Performance

From test run on `exemple.json`:

- **Total Violations**: 38 detected
- **Processing Mode**: Hybrid AI+Rules
- **Confidence Range**: 0-100%
- **API Calls**: ~50+ (with caching)
- **Categories**: 5 (Structure, General, Securities, Performance, Prospectus)

---

## 🎓 Key Concepts

### Confidence Scores

- **0-59%**: Low confidence - Needs human review
- **60-84%**: Medium confidence - AI detected variation
- **85-100%**: High confidence - Verified by both AI and rules

### Status Types

- `VERIFIED_BY_BOTH`: Both AI and rules agree
- `AI_DETECTED_VARIATION`: AI found something rules missed
- `FALSE_POSITIVE_FILTERED`: Rules flagged but AI cleared
- `NEEDS_REVIEW`: Below confidence threshold
- `COMPLIANT`: No violations

---

## 🔮 Future Enhancements

- Pattern learning from AI findings
- Multi-model AI support
- Real-time monitoring dashboard
- Parallel batch processing
- Custom rule definitions
- PDF/HTML report generation

---

## 📞 Support

For issues or questions:

1. Check [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
2. Review `hybrid_config.json` settings
3. Check logs for error messages
4. Test with `--rules-only` to isolate AI issues
5. Verify API keys and network connectivity

---

## 📄 License

Proprietary - All rights reserved

---

## ✅ Status

**Production Ready** - All features implemented and tested

- ✅ Hybrid AI+Rules architecture
- ✅ 100% backward compatibility
- ✅ Comprehensive compliance checks
- ✅ Intelligent caching
- ✅ Error handling & fallback
- ✅ Configuration system
- ✅ Complete documentation

---

**Last Updated**: 2025-01-18  
**Version**: 1.0  
**Status**: Production Ready ✅
