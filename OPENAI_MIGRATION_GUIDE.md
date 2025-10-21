# OpenAI Migration Guide

## Overview
This project has been migrated from **Ollama** to **OpenAI API** for enhanced AI capabilities and better reliability.

## What Changed

### 1. New Files Created
- **`.env`**: Contains your OpenAI API key and environment configuration
- **`openai_integration.py`**: New module that replaces `ollama_simple_integration.py`
- **`test_openai_integration.py`**: Test script to verify OpenAI integration

### 2. Files Updated
- **`app.py`**: All Ollama imports replaced with OpenAI imports
- **`real_vendor_extraction.py`**: Updated to use OpenAI for vendor extraction
- **`integrate_advanced_revenue_system.py`**: Updated health checks to use OpenAI
- **`requirements.txt`**: OpenAI dependency highlighted, Ollama commented out

### 3. Key Changes
- All `ollama_simple_integration` imports → `openai_integration`
- `simple_ollama()` → `simple_openai()` (with backward compatibility alias)
- `OllamaSimpleIntegration` → `OpenAIIntegration` (with backward compatibility alias)
- Model names: `llama3.2:3b` → `gpt-4o-mini` (cost-effective model)

## Setup Instructions

### 1. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Configure Environment
Your `.env` file has been created with your API key:
\`\`\`
OPENAI_API_KEY=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
ENVIRONMENT=LOCAL
\`\`\`

**⚠️ IMPORTANT**: Never commit the `.env` file to git. It's already in `.gitignore`.

### 3. Test the Integration
\`\`\`bash
python test_openai_integration.py
\`\`\`

This will test:
- API key configuration
- Basic API calls
- Transaction categorization
- Vendor extraction

### 4. Run Your Application
\`\`\`bash
python app.py
\`\`\`

## Features & Benefits

### OpenAI Advantages over Ollama
1. **Better Accuracy**: GPT models provide more accurate categorization and vendor extraction
2. **Faster Processing**: Cloud-based processing with optimized infrastructure
3. **No Local Setup**: No need to run Ollama server locally or on EC2
4. **Scalability**: Handles large batches efficiently
5. **Reliability**: 99.9% uptime with automatic failover

### Cost Considerations
- Using **gpt-4o-mini** (most cost-effective model)
- Typical costs: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- For 1000 transactions: ~$0.05-0.10
- For 10,000 transactions: ~$0.50-1.00

### Model Options
The integration supports multiple models:
- **`gpt-4o-mini`**: Default, best cost/performance ratio (recommended)
- **`gpt-4o`**: Most accurate, higher cost
- **`gpt-3.5-turbo`**: Fastest, lower accuracy

To change models, update the `default_model` in `openai_integration.py`

## API Usage

### Basic Usage
\`\`\`python
from openai_integration import simple_openai

# Simple text generation
response = simple_openai("Analyze this transaction", max_tokens=100)
\`\`\`

### Transaction Categorization
\`\`\`python
from openai_integration import openai_integration

descriptions = ["Payment to vendor", "Loan repayment"]
categories = openai_integration.categorize_transactions(descriptions)
\`\`\`

### Vendor Extraction
\`\`\`python
from openai_integration import openai_integration

descriptions = ["Payment to Tata Steel"]
vendors = openai_integration.extract_vendors_for_transactions(descriptions)
\`\`\`

### Pattern Analysis
\`\`\`python
from openai_integration import openai_integration

data = [{"amount": 1000, "description": "Payment"}]
analysis = openai_integration.analyze_patterns(data)
\`\`\`

## Backward Compatibility

The integration maintains backward compatibility with existing code:
- `simple_ollama()` still works (aliased to `simple_openai()`)
- `OllamaSimpleIntegration` still works (aliased to `OpenAIIntegration`)
- All function signatures remain the same

## Troubleshooting

### API Key Issues
If you see "No OpenAI API key provided":
1. Check `.env` file exists in project root
2. Verify `OPENAI_API_KEY` is set correctly
3. Restart your application

### Import Errors
If you see "Cannot import openai_integration":
1. Ensure `openai_integration.py` is in project root
2. Run: `pip install openai python-dotenv`
3. Check for syntax errors in the module

### API Rate Limits
If you hit rate limits:
1. Add delays between requests
2. Use batch processing (already implemented)
3. Upgrade OpenAI plan if needed

### Connection Issues
If API calls fail:
1. Check internet connection
2. Verify API key is valid
3. Check OpenAI status: https://status.openai.com/

## Monitoring & Debugging

### Enable Debug Logging
\`\`\`python
import logging
logging.basicConfig(level=logging.DEBUG)
\`\`\`

### Check Health Status
\`\`\`python
from openai_integration import openai_integration
status = openai_integration.get_health_status()
print(status)
\`\`\`

### View API Usage
- Visit: https://platform.openai.com/usage
- Monitor costs and request counts

## Security Best Practices

1. **Never commit `.env` file** - Already in `.gitignore`
2. **Rotate API keys regularly** - Every 3-6 months
3. **Use environment variables** - Already configured
4. **Monitor usage** - Set up billing alerts on OpenAI dashboard
5. **Limit API key permissions** - Use restricted keys if possible

## Migration from Ollama

If you need to revert or maintain both systems:

### OpenAI Only Mode (Current Configuration)
This project now uses **OpenAI exclusively** with no fallbacks. If OpenAI is unavailable, the application will fail with an error message.

### Revert to Ollama
1. Restore old imports in files
2. Uncomment `ollama==0.5.1` in requirements.txt
3. Install: `pip install ollama`
4. Start Ollama server

## Support & Resources

### OpenAI Documentation
- API Reference: https://platform.openai.com/docs/api-reference
- Guides: https://platform.openai.com/docs/guides
- Pricing: https://openai.com/pricing

### Project Resources
- Test Script: `python test_openai_integration.py`
- Integration Module: `openai_integration.py`
- Configuration: `.env`

### Common Issues
1. **Slow responses**: Increase timeout or reduce max_tokens
2. **High costs**: Switch to gpt-4o-mini or reduce batch sizes
3. **Rate limits**: Add exponential backoff (already implemented)

## Performance Optimization

### Tips for Better Performance
1. **Batch Processing**: Already implemented for large datasets
2. **Caching**: Consider caching similar requests
3. **Async Processing**: Use async for concurrent requests
4. **Model Selection**: Use gpt-4o-mini for most tasks
5. **Token Optimization**: Keep prompts concise

### Current Optimizations
- ✅ Automatic retry with exponential backoff
- ✅ Batch processing for vendor extraction
- ✅ Response caching in vendor extraction
- ✅ Adaptive timeout based on prompt complexity
- ✅ Error handling and fallbacks

## Conclusion

Your project is now fully migrated to OpenAI! The integration provides:
- 🚀 Better accuracy and reliability
- ⚡ Faster processing
- 🔧 Easier maintenance (no local server needed)
- 📊 Better scalability

If you have any issues, run the test script or check the troubleshooting section above.

---

**Last Updated**: October 11, 2025
**Migration Version**: 1.0
**OpenAI Package**: 1.93.1
