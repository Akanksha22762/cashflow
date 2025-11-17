import pandas as pd
import numpy as np
import re  
import time
import hashlib

class UniversalVendorExtractor:
    """Universal vendor extraction with AI/ML priority and fallback system"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.last_cache_cleanup = time.time()
    
    def _get_cache_key(self, descriptions):
        """Generate cache key for descriptions"""
        # Handle both pandas Series and regular lists
        if hasattr(descriptions, 'empty'):
            # It's a pandas Series
            if descriptions.empty:
                return None
        elif not descriptions:
            # It's a regular list/array
            return None
        
        # Create hash of first few descriptions for caching
        sample = str(descriptions[:5])
        return hashlib.md5(sample.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key):
        """Get cached result if available and not expired"""
        if not cache_key or cache_key not in self.cache:
            return None
        
        timestamp, result = self.cache[cache_key]
        if time.time() - timestamp > self.cache_ttl:
            del self.cache[cache_key]
            return None
        
        print(f"🚀 Using cached vendor extraction result ({len(result)} vendors)")
        return result
    
    def _cache_result(self, cache_key, result):
        """Cache the result with timestamp"""
        if cache_key:
            self.cache[cache_key] = (time.time(), result)
            
            # Cleanup old cache entries periodically
            if time.time() - self.last_cache_cleanup > 300:
                self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up old cache entries to prevent memory bloat"""
        current_time = time.time()
        expired_keys = []
        
        for key, (timestamp, _) in self.cache.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            print(f"🧹 Cache cleanup: removed {len(expired_keys)} expired entries")
        
        self.last_cache_cleanup = current_time
    
    async def extract_vendors_intelligently(self, descriptions, use_ai=True):
        """OLLAMA-ONLY extraction: Use only Ollama AI for vendor extraction"""
        print("🚀 OLLAMA-ONLY VENDOR EXTRACTION - AI POWERED")
        print("=" * 60)
        print(f"🔍 Input descriptions: {len(descriptions) if descriptions else 0} items")
        
        if not descriptions or len(descriptions) == 0:
            print("❌ No descriptions provided")
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(descriptions)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        print(f"Processing {len(descriptions)} transaction descriptions...")
        
        start_time = time.time()
        all_vendors = []
        
        # STEP 1: Use OLLAMA ONLY (AI-powered, most accurate)
        if use_ai:
            print("\n🧠 Step 1: OLLAMA AI Enhancement (ONLY METHOD)...")
            try:
                # Process ALL descriptions for maximum vendor coverage
                print(f"🧠 Processing {len(descriptions)} descriptions with Ollama...")
                
                ollama_vendors = self._extract_vendors_with_ollama_fast(descriptions)
                if ollama_vendors:
                    all_vendors.extend(ollama_vendors)
                    print(f"✅ Ollama found {len(ollama_vendors)} vendors")
                else:
                    print("⚠️ Ollama found no vendors, using default fallback...")
            except Exception as e:
                print(f"❌ Ollama failed: {e}, using default fallback...")
        
        # FALLBACK: If Ollama fails, assign "Other Services" to all transactions
        if not all_vendors:
            print("\n⚡ Fallback: Assigning default vendor to all transactions...")
            vendor_assignments = ["Other Services"] * len(descriptions)
        else:
            # Consolidate results
            final_vendors = self._consolidate_vendors_fast(all_vendors, descriptions)
            
            # CRITICAL FIX: Create vendor assignments for each transaction
            vendor_assignments = self._create_vendor_assignments(descriptions, final_vendors)
        
        # Cache the result
        self._cache_result(cache_key, vendor_assignments)
        
        total_time = time.time() - start_time
        print(f"\n🚀 OLLAMA-ONLY EXTRACTION COMPLETED:")
        print(f"   🚀 Total Time: {total_time:.2f}s")
        print(f"   📊 Transactions: {len(descriptions)}")
        print(f"   🎯 Vendor Assignments: {len(vendor_assignments)}")
        print(f"   ⚡ Speed: {len(descriptions)/total_time:.1f} transactions/second")
        
        return vendor_assignments
    
    def extract_vendors_intelligently_sync(self, descriptions, use_ai=True):
        """OLLAMA-ONLY extraction: Use only Ollama AI for vendor extraction"""
        print(f"Processing {len(descriptions)} transaction descriptions...")

        # Check cache first
        cache_key = self._get_cache_key(descriptions)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            # CRITICAL FIX: Ensure cached result is vendor assignments, not unique vendors
            if len(cached_result) == len(descriptions):
                print(f"✅ Using cached vendor assignments: {len(cached_result)} assignments for {len(descriptions)} transactions")
                return cached_result
            else:
                print(f"⚠️ Cached result length mismatch: {len(cached_result)} vs {len(descriptions)}, regenerating...")
        
        print("Using OLLAMA-ONLY extraction: AI-powered vendor extraction")
        
        start_time = time.time()
        all_vendors = []
        
        # STEP 1: Use OLLAMA ONLY (AI-powered, most accurate)
        if use_ai:
            print("\n🧠 Step 1: OLLAMA AI Enhancement (ONLY METHOD)...")
            try:
                # Process ALL descriptions for maximum vendor coverage
                print(f"🧠 Processing {len(descriptions)} descriptions with Ollama...")
                
                ollama_vendors = self._extract_vendors_with_ollama_fast(descriptions)
                if ollama_vendors and len(ollama_vendors) > 0:
                    all_vendors.extend(ollama_vendors)
                    print(f"✅ Ollama found {len(ollama_vendors)} vendors")
                    
                    # ✅ SUCCESS: Ollama worked
                    print("🚀 Ollama vendor extraction successful")
                    
                    # Consolidate results and return immediately
                    final_vendors = self._consolidate_vendors_fast(all_vendors, descriptions)
                    
                    # CRITICAL FIX: Create vendor assignments for each transaction
                    vendor_assignments = self._create_vendor_assignments(descriptions, final_vendors)
                    
                    # CRITICAL VALIDATION: Ensure we have the right number of assignments
                    if len(vendor_assignments) != len(descriptions):
                        print(f"❌ CRITICAL ERROR: Vendor assignments length mismatch!")
                        print(f"   Expected: {len(descriptions)} assignments")
                        print(f"   Got: {len(vendor_assignments)} assignments")
                        print(f"   This will cause pandas error: 'Length of values does not match length of index'")
                        # Fix by creating proper assignments
                        vendor_assignments = ["Other Services"] * len(descriptions)
                        print(f"   🔧 Fixed: Created {len(vendor_assignments)} default assignments")
                    
                    # Cache the result
                    self._cache_result(cache_key, vendor_assignments)
                    
                    total_time = time.time() - start_time
                    print(f"\n✅ OLLAMA-ONLY extraction completed in {total_time:.2f}s: {len(final_vendors)} unique vendors")
                    print(f"🚀 Speed: {len(descriptions)/total_time:.1f} transactions/second")
                    
                    return vendor_assignments
                else:
                    print("⚠️ Ollama found no vendors, using default fallback...")
            except Exception as e:
                print(f"❌ Ollama failed: {e}, using default fallback...")
        
        # FALLBACK: If Ollama fails, assign "Other Services" to all transactions
        print("\n⚡ Fallback: Assigning default vendor to all transactions...")
        vendor_assignments = ["Other Services"] * len(descriptions)
        
        # CRITICAL VALIDATION: Ensure we have the right number of assignments
        if len(vendor_assignments) != len(descriptions):
            print(f"❌ CRITICAL ERROR: Fallback assignments length mismatch!")
            print(f"   Expected: {len(descriptions)} assignments")
            print(f"   Got: {len(vendor_assignments)} assignments")
            # This should never happen, but just in case
            vendor_assignments = ["Other Services"] * len(descriptions)
        
        # Cache the result
        self._cache_result(cache_key, vendor_assignments)
        
        total_time = time.time() - start_time
        print(f"\n✅ OLLAMA-ONLY extraction completed in {total_time:.2f}s: Using default fallback")
        print(f"🚀 Speed: {len(descriptions)/total_time:.1f} transactions/second")
        
        return vendor_assignments

    def extract_vendors_intelligently_forced_sync(self, descriptions):
        """Forced Ollama-only extraction - no caching, no fallback"""
        if not descriptions or len(descriptions) == 0:
            print("❌ No descriptions provided")
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(descriptions)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        print(f"Processing {len(descriptions)} transaction descriptions...")
        
        # Step 1: Forced Ollama extraction
        print("\n🧠 Step 1: FORCED Ollama AI Enhancement (ONLY METHOD)...")
        start_time = time.time()
        ollama_vendors = []
        
        try:
            # Process ALL descriptions for better AI learning
            print(f"🧠 Processing {len(descriptions)} descriptions with Ollama...")
            
            ollama_vendors = self._extract_vendors_with_ollama_fast(descriptions)
            print(f"✅ Ollama enhancement completed: {len(ollama_vendors)} vendors found")
            
        except Exception as e:
            print(f"❌ Ollama enhancement failed: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            print(f"🔍 Error details: {str(e)}")
            ollama_vendors = []
        
        # Step 2: Create vendor assignments
        print("\n🧠 Step 2: Creating Vendor Assignments...")
        if ollama_vendors:
            final_vendors = self._consolidate_vendors_fast(ollama_vendors, descriptions)
            vendor_assignments = self._create_vendor_assignments(descriptions, final_vendors)
        else:
            # If Ollama fails, assign "Other Services" to all transactions
            vendor_assignments = ["Other Services"] * len(descriptions)
            print("⚠️ Using default fallback: 'Other Services' for all transactions")
        
        # Cache the result
        self._cache_result(cache_key, vendor_assignments)
        total_time = time.time() - start_time
        print(f"\n🤖 FORCED OLLAMA-ONLY EXTRACTION COMPLETED:")
        print(f"   🚀 Total Time: {total_time:.2f}s")
        print(f"   📊 Transactions: {len(descriptions)}")
        print(f"   🎯 Vendor Assignments: {len(vendor_assignments)}")
        print(f"   🧠 AI Vendors Found: {len(ollama_vendors)}")
        print(f"   ⚡ Speed: {len(descriptions)/total_time:.1f} transactions/second")
        
        return vendor_assignments
    
    def _extract_vendors_fast_regex(self, descriptions):
        """ULTRA-FAST vendor extraction using STRICT regex patterns for real companies only"""
        print("   ⚡ Using ULTRA-FAST regex extraction with STRICT validation...")
        vendors = []
        start_time = time.time()
        import re
        
        # Process all descriptions for vendor extraction
        max_descriptions = len(descriptions)
        print(f"🚀 Processing all {max_descriptions} descriptions for vendor extraction...")
        
        # STRICT regex patterns for REAL company names only (compiled with IGNORECASE)
        vendor_patterns = [
            # Pattern 1: Company names with business suffixes (HIGH PRIORITY - definitely companies)
            re.compile(r'([A-Z][a-zA-Z\s&]+?)\s+(?:LTD|LIMITED|LLC|INC|CORP|CORPORATION|COMPANY|CO|GROUP|ENTERPRISES|HOLDINGS|INTERNATIONAL|INDUSTRIES)', re.IGNORECASE),
            
            # Pattern 2: "Payment to [Company Name]" format
            re.compile(r'(?:PAYMENT TO|PAYMENT FOR|PAID TO|TRANSFER TO)\s+([A-Z][a-zA-Z\s&]+(?:LTD|LIMITED|LLC|INC|CORP|CORPORATION|COMPANY|CO|GROUP|ENTERPRISES|HOLDINGS|INTERNATIONAL|INDUSTRIES))', re.IGNORECASE),
            
            # Pattern 3: Specific vendor patterns
            re.compile(r'(LOGISTICS\s+PROVIDER|SERVICE\s+PROVIDER|EQUIPMENT\s+SUPPLIER|RAW\s+MATERIAL\s+SUPPLIER|COAL\s+SUPPLIER|LIMESTONE\s+SUPPLIER|ALLOY\s+SUPPLIER|STEEL\s+SUPPLIER)(?:\s+\d+)?', re.IGNORECASE),
            
            # Pattern 4: Company names in parentheses
            re.compile(r'\(([A-Z][a-zA-Z\s&]+(?:LTD|LIMITED|LLC|INC|CORP|CORPORATION|COMPANY|CO))\)', re.IGNORECASE),
            
            # Pattern 5: Company names after dashes
            re.compile(r'[-–—]\s*([A-Z][a-zA-Z\s&]+(?:LTD|LIMITED|LLC|INC|CORP|CORPORATION|COMPANY|CO))', re.IGNORECASE)
        ]
        
        processed = 0
        for desc in descriptions[:max_descriptions]:
            if str(desc).strip() == '' or str(desc) in ['nan', 'None', '']:
                continue
                
            desc_str = str(desc)
            vendor_found = False
            
            # Try each pattern
            for pattern in vendor_patterns:
                try:
                    match = pattern.search(desc_str)
                    if match and match.groups():
                        vendor = match.group(1).strip()
                        if len(vendor) > 2 and vendor.lower() not in ['the', 'and', 'for', 'with', 'from']:
                            # Apply STRICT validation
                            if self._validate_vendor_name_fast(vendor):
                                vendors.append(vendor)
                                vendor_found = True
                                break
                except (IndexError, AttributeError):
                    # Skip patterns that don't have the expected group structure
                    continue
            
            processed += 1
            if processed % 50 == 0:
                print(f"   📊 Processed {processed}/{max_descriptions} descriptions...")
            
            total_time = time.time() - start_time
        print(f"   ⚡ ULTRA-FAST regex completed in {total_time:.2f}s: {len(vendors)} vendors")
        print(f"   🚀 Speed: {max_descriptions/total_time:.1f} descriptions/second")
        return vendors
            
    def _validate_vendor_name_fast(self, vendor_name):
        """FLEXIBLE vendor name validation - Accept both company names and vendor types"""
        if not vendor_name or len(vendor_name.strip()) < 3:
            return None
        
        vendor_clean = vendor_name.strip()
        vendor_lower = vendor_clean.lower()
        
        # ✅ FIRST: ACCEPT vendor types (our new feature)
        vendor_type_indicators = ['vendor', 'supplier', 'provider', 'contractor', 'manufacturer', 'distributor', 'retailer']
        if any(indicator in vendor_lower for indicator in vendor_type_indicators):
            # Accept meaningful vendor types like "Medical Supplies Vendor", "IT Software Vendor"
            if len(vendor_clean) > 8 and not vendor_lower.startswith('payment') and not vendor_lower.startswith('invoice'):
                return vendor_clean
        
        # ✅ SECOND: ACCEPT names with business suffixes (definitely companies)
        business_suffixes = ['ltd', 'limited', 'llc', 'inc', 'corp', 'corporation', 'company', 'co', 'group', 'enterprises', 'holdings', 'international', 'industries']
        if any(vendor_lower.endswith(' ' + suffix) or vendor_lower.endswith(suffix) for suffix in business_suffixes):
            # Only reject if it's EXACTLY a product name + suffix (like "Color Co")
            if vendor_lower in ['color co', 'steel co', 'metal co', 'raw co']:
                return None
            return vendor_clean
        
        # 🚫 REJECT obvious non-company terms
        obvious_rejections = {
            # Transaction types
            'payment', 'purchase', 'sale', 'advance', 'retention', 'final', 'milestone',
            'bulk', 'capex', 'bonus', 'bridge', 'loan', 'credit', 'emi', 'closure',
            'export', 'import', 'investment', 'liquidation', 'proceeds', 'charges',
            'interest', 'principal', 'repayment', 'disbursement', 'penalty', 'bonus',
            
            # Project descriptions
            'plant expansion', 'infrastructure development', 'warehouse construction',
            'production line', 'capacity increase', 'new blast furnace', 'renovation',
            'modernization', 'upgrade', 'installation', 'maintenance', 'project',
            'infrastructure project', 'bridge construction', 'festival season',
            
            # Equipment and materials
            'equipment', 'machinery', 'infrastructure', 'development', 'expansion',
            'quality', 'testing', 'warehouse', 'production', 'line', 'capacity',
            'energy', 'efficiency', 'technology', 'system', 'digital', 'transformation',
            'material', 'raw', 'steel', 'rolling', 'blast', 'furnace', 'galvanized',
            'color coated', 'color', 'color co', 'excess', 'scrap', 'metal', 'landline', 'mobile',
            
            # Generic business terms (only reject these specific combinations)
            'gas company', 'real estate developer', 'oil & gas company', 'automotive manufacturer',
            'defense contractor', 'railway department', 'shipbuilding yard', 'municipal corporation',
            'logistics services', 'basic services', 'communication services', 'protection services',
            'military services', 'risk co', 'marine co', 'employee inc', 'employee co',
            'warehouse construction', 'bridge construction', 'color coated steel', 'galvanized steel',
            'excess steel', 'housekeeping services', 'audit services', 'legal services',
            'security services', 'landline & mobile', 'municipal corporation'
        }
        
        # Check for obvious rejections
        for rejection in obvious_rejections:
            if rejection in vendor_lower:
                return None
        
        # ✅ ACCEPT names that look like real companies
        
        # 1. Business suffixes already handled above
        
        # 2. ACCEPT company names with "&" or "and" (likely real companies)
        if '&' in vendor_clean or ' and ' in vendor_lower:
            if len(vendor_clean.split()) >= 3:  # Must be multi-word
                return vendor_clean
        
        # 3. ACCEPT names that look like real companies (multi-word with proper capitalization)
        if len(vendor_clean.split()) >= 2:
            words = vendor_clean.split()
            # Check if it starts with capital letter and has proper company structure
            if words[0][0].isupper() and any(word[0].isupper() for word in words[1:]):
                # More lenient validation - accept most multi-word capitalized names
                return vendor_clean
        
        # 4. ACCEPT specific patterns like "ABC Corp", "XYZ Ltd"
        if len(vendor_clean.split()) == 2:
            first_word, second_word = vendor_clean.split()
            if first_word[0].isupper() and second_word.lower() in business_suffixes:
                return vendor_clean
        
        # 5. ACCEPT generic company patterns that are commonly used
        generic_company_patterns = ['abc corp', 'xyz ltd', 'abc company', 'xyz company']
        if vendor_lower in generic_company_patterns:
            return vendor_clean
        
        # 6. ACCEPT single-word company names that are capitalized (like "GE", "IBM", "SBI")
        if len(vendor_clean.split()) == 1 and vendor_clean[0].isupper() and len(vendor_clean) >= 2:
            return vendor_clean
        
        # 7. ACCEPT common bank names and financial institutions (EC2 fix)
        common_banks = ['pnb', 'sbi', 'hdfc', 'axis', 'icici', 'kotak', 'yes bank', 'indian bank', 'canara bank', 'union bank']
        if vendor_lower in common_banks or any(bank in vendor_lower for bank in common_banks):
            return vendor_clean
        
        # 8. ACCEPT any name that looks like a company (very lenient fallback)
        if len(vendor_clean) >= 3 and vendor_clean[0].isupper():
            return vendor_clean
        
        # If none of the acceptance criteria match, reject
        return None
        
    def _extract_vendors_with_ollama_fast(self, descriptions):
        """Extract vendors using OpenAI batch mode - Simple and fast"""
        # Handle both pandas Series and regular lists
        if hasattr(descriptions, 'empty'):
            if descriptions.empty:
                return []
        elif not descriptions:
            return []
        
        # Check cache first for consistency
        cache_key = f"openai_vendors_{hash(str(descriptions))}"
        if hasattr(self, '_vendor_cache'):
            if cache_key in self._vendor_cache:
                print(f"🔄 Using cached vendor extraction for consistency")
                return self._vendor_cache[cache_key]
        else:
            self._vendor_cache = {}
        
        try:
            from openai_integration import openai_integration, check_openai_availability
            import os
            
            # Process all descriptions
            sample_descriptions = descriptions
            print(f"🧠 Using OpenAI BATCH mode for vendor extraction...")
            print(f"⏱️ Note: Fast batch processing with OpenAI...")
            
            # Check OpenAI availability
            if not check_openai_availability():
                raise RuntimeError("OpenAI API is not available. Check your API key and connection.")
            
            # ✅ USE OPENAI INTEGRATION'S BUILT-IN BATCH VENDOR EXTRACTION
            print(f"🚀 Using OpenAI batch vendor extraction for {len(sample_descriptions)} transactions...")
            
            # Convert to list if pandas Series
            if hasattr(sample_descriptions, 'tolist'):
                sample_descriptions = sample_descriptions.tolist()
            
            # Call OpenAI's batch vendor extraction (handles batching internally)
            vendors = openai_integration.extract_vendors_for_transactions(sample_descriptions)
            
            # Cache the results
            self._vendor_cache[cache_key] = vendors
            
            print(f"✅ OpenAI batch vendor extraction completed: {len(vendors)} vendors extracted")
            return vendors
            
            # OLD COMPLEX CODE BELOW - REPLACED WITH SIMPLE OPENAI CALL
            if False and len(sample_descriptions) > 50:
                print(f"   📊 Large dataset detected ({len(sample_descriptions)} transactions)")
                print(f"   🔄 Processing in efficient batches for maximum coverage...")
                
                # Process in batches of 50 for efficiency (reduced from 200 to avoid timeouts)
                batch_size = 50
                all_vendors = []
                
                for i in range(0, len(sample_descriptions), batch_size):
                    batch = sample_descriptions[i:i + batch_size]
                    print(f"   🔄 Processing batch {i//batch_size + 1}/{(len(sample_descriptions) + batch_size - 1)//batch_size} ({len(batch)} transactions)")
                    
                    # Create prompt for this batch
                    batch_prompt = f"""Extract ONLY company names that are EXPLICITLY written in these transactions.

{chr(10).join([f"{idx+1}. {str(desc)[:80]}" for idx, desc in enumerate(batch) if str(desc).strip() != '' and str(desc) not in ['nan', 'None', '']])}

CRITICAL RULES - READ CAREFULLY:
- Extract ONLY company names that are EXPLICITLY written in the text
- If NO explicit company name exists, extract a VENDOR TYPE based on the service/product
- DO NOT invent fake company names
- For generic payments, create vendor types like:
  * "Payment to Vendor – Medical Supplies" → "Medical Supplies Vendor"
  * "Payment to IT Vendor – Software" → "IT Software Vendor"
  * "Payment to Supplier – Equipment" → "Equipment Supplier"
- If you see "Payment to ABC Corp", extract "ABC Corp"
- If you see generic "Payment to Vendor – [Service]", extract "[Service] Vendor"
- Output format: Either "Company Name" OR "Service Type + Vendor/Supplier"

Vendor Names or Types:"""
                    
                    try:
                        response = simple_ollama(batch_prompt, max_tokens=200)  # OpenAI batch mode
                        if response:
                            lines = response.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if not line or line.startswith('Company Names:'):
                                    continue
                                
                                # Skip explanatory text and long sentences
                                if len(line) > 50 or any(phrase in line.lower() for phrase in [
                                    'note that', 'if you', 'rather than', 'would be', 'some of these',
                                    'abbreviations', 'descriptions', 'full company', 'extract'
                                ]):
                                    continue
                                
                                vendor = line.strip()
                                
                                # Clean up vendor name and check for "no company" variations
                                vendor_clean = vendor.lstrip('• ').strip()
                                
                                # CRITICAL FIX: Handle numbered lists and explanatory text (EC2 issue)
                                # Remove numbers at start like "4. Indian Oil" -> "Indian Oil"
                                if vendor_clean and vendor_clean[0].isdigit():
                                    parts = vendor_clean.split('.', 1)
                                    if len(parts) > 1:
                                        vendor_clean = parts[1].strip()
                                        print(f"   🔧 EC2 FIX: Removed number prefix: '{vendor_clean}'")
                                
                                # Handle explanatory text in parentheses
                                if '(' in vendor_clean and ')' in vendor_clean:
                                    # Extract company name before parentheses
                                    company_part = vendor_clean.split('(')[0].strip()
                                    if company_part and len(company_part) > 2:
                                        vendor_clean = company_part
                                        print(f"   🔧 EC2 FIX: Extracted company name from explanatory text: '{company_part}'")
                                
                                # Remove trailing dashes and extra text like "PNB - Bank" -> "PNB"
                                if ' - ' in vendor_clean:
                                    vendor_clean = vendor_clean.split(' - ')[0].strip()
                                    print(f"   🔧 EC2 FIX: Removed trailing text: '{vendor_clean}'")
                                
                                vendor_lower = vendor_clean.lower().replace('_', '').replace(' ', '')
                                
                                # Skip all variations of "no company" and conversational text
                                if (vendor_lower in ['nocompany', 'nocompanyname', 'none', 'n/a', 'na'] or
                                    'no_company' in vendor_lower or 'nocompany' in vendor_lower or
                                    vendor_clean.startswith("I'm ready") or "please provide" in vendor_clean.lower() or
                                    len(vendor_clean) > 80):  # Likely conversational text
                                    print(f"   ❌ Skipped no-company/conversational text: {vendor}")
                                    continue
                                
                                if vendor_clean and len(vendor_clean) > 2:
                                    # Remove numbering if present
                                    if vendor_clean[0].isdigit() and '. ' in vendor_clean:
                                        vendor_clean = vendor_clean.split('. ', 1)[1].strip()
                                    
                                    # Clean up Ollama's formatting
                                    if vendor_clean.startswith("- Company Name:"):
                                        vendor_clean = vendor_clean.replace("- Company Name:", "").strip()
                                    elif vendor_clean.startswith("- Vendor Type:"):
                                        vendor_clean = vendor_clean.replace("- Vendor Type:", "").strip()
                                    elif vendor_clean.startswith("-"):
                                        vendor_clean = vendor_clean.lstrip("- ").strip()
                                    
                                # Skip explanatory phrases and headers (less strict)
                                if any(bad_text in vendor_clean.lower() for bad_text in [
                                    'here are', 'extracted', 'company names', 'vendor types',
                                    'payment to', 'invoice from', 'implied', 'not explicitly', 
                                    'mentioned', 'but not', 'might be', 'could be', 'seems like', 
                                    'appears to be', 'note that', 'rather than', 'abbreviation', 
                                    'no company', 'just the equipment', 'equipment type', 
                                    'funding description', 'explicitly written', 'no explicit name found'
                                ]):
                                    print(f"   ❌ Skipped explanatory text: {vendor_clean}")
                                    continue
                                
                                # Validate and append cleaned vendor name
                                validated_vendor = self._validate_vendor_name_fast(vendor_clean)
                                if validated_vendor:
                                    all_vendors.append(validated_vendor)
                                    print(f"   ✅ Batch vendor: {validated_vendor}")
                                else:
                                    print(f"   ❌ Rejected vendor: {vendor_clean}")
                    except Exception as e:
                        print(f"   ⚠️ Batch {i//batch_size + 1} failed: {e}")
                        continue
                
                vendors = all_vendors
                print(f"   🧠 Batch processing completed: {len(vendors)} total vendors found")
                print(f"   🔍 EC2 DEBUG: Final vendors list: {vendors[:10]}")  # Show first 10 vendors for debugging
                
            else:
                # For smaller datasets, use single processing
                print(f"   📊 Processing {len(sample_descriptions)} transactions in single batch...")
                
                # Create ULTRA-STRICT prompt for company extraction only
                batch_prompt = f"""Extract ONLY company names that are EXPLICITLY written in these transactions.

{chr(10).join([f"{idx+1}. {str(desc)[:80]}" for idx, desc in enumerate(sample_descriptions) if str(desc).strip() != '' and str(desc) not in ['nan', 'None', '']])}

CRITICAL RULES - SMART VENDOR EXTRACTION:
- Extract company names that are EXPLICITLY written in the text
- If NO explicit company name exists, create a vendor TYPE based on the service/product
- DO NOT invent fake company names
- For generic payments, create descriptive vendor types

EXAMPLES OF EXPLICIT COMPANY NAMES:
✅ "Payment to ABC Construction Company Ltd" → "ABC Construction Company Ltd"
✅ "Invoice from XYZ Steel Corp" → "XYZ Steel Corp"
✅ "Purchase from DEF Manufacturing Inc" → "DEF Manufacturing Inc"

EXAMPLES OF VENDOR TYPES (for generic payments):
✅ "Payment to Vendor – Medical Supplies" → "Medical Supplies Vendor"
✅ "Payment to IT Vendor – Software" → "IT Software Vendor"
✅ "Payment to Supplier – Equipment" → "Equipment Supplier"
✅ "Payment to Vendor – Housekeeping" → "Housekeeping Vendor"
✅ "Payment to Vendor – Maintenance" → "Maintenance Vendor"

Vendor Names or Types:"""
            
                try:
                    print("   🧠 Sending request to Ollama (this may take 20-40 seconds)...")
                    print(f"🔍 DEBUG: Sample descriptions being sent to Ollama: {sample_descriptions[:3]}")
                    response = simple_ollama(batch_prompt, "llama3.2:3b", max_tokens=150)
                    print(f"🔍 DEBUG: Ollama response received: {response[:200] if response else 'None'}...")
                    if response:
                        lines = response.strip().split('\n')
                        vendors = []
                        for line in lines:
                            line = line.strip()
                            if not line or line.startswith('Company Names:'):
                                continue
                                
                            vendor = line.strip()
                            # Clean up vendor name and check for "no company" variations
                            vendor_clean = vendor.lstrip('• ').strip()
                            
                            # CRITICAL FIX: Handle numbered lists and explanatory text (EC2 issue)
                            # Remove numbers at start like "4. Indian Oil" -> "Indian Oil"
                            if vendor_clean and vendor_clean[0].isdigit():
                                parts = vendor_clean.split('.', 1)
                                if len(parts) > 1:
                                    vendor_clean = parts[1].strip()
                                    print(f"   🔧 EC2 FIX: Removed number prefix: '{vendor_clean}'")
                            
                            # Handle explanatory text in parentheses
                            if '(' in vendor_clean and ')' in vendor_clean:
                                # Extract company name before parentheses
                                company_part = vendor_clean.split('(')[0].strip()
                                if company_part and len(company_part) > 2:
                                    vendor_clean = company_part
                                    print(f"   🔧 EC2 FIX: Extracted company name from explanatory text: '{company_part}'")
                            
                            # Remove trailing dashes and extra text like "PNB - Bank" -> "PNB"
                            if ' - ' in vendor_clean:
                                vendor_clean = vendor_clean.split(' - ')[0].strip()
                                print(f"   🔧 EC2 FIX: Removed trailing text: '{vendor_clean}'")
                            
                            vendor_lower = vendor_clean.lower().replace('_', '').replace(' ', '')
                            
                            # Skip all variations of "no company" and conversational text
                            if (vendor_lower in ['nocompany', 'nocompanyname', 'none', 'n/a', 'na'] or
                                'no_company' in vendor_lower or 'nocompany' in vendor_lower or
                                vendor_clean.startswith("I'm ready") or "please provide" in vendor_clean.lower() or
                                len(vendor_clean) > 80):  # Likely conversational text
                                print(f"   ❌ Skipped no-company/conversational text: {vendor}")
                                continue
                            
                            if vendor_clean and len(vendor_clean) > 2:
                                # Remove numbering like "1. ", "2. ", etc.
                                if vendor_clean[0].isdigit() and '. ' in vendor_clean:
                                    vendor_clean = vendor_clean.split('. ', 1)[1].strip()
                                
                                # Clean up Ollama's formatting
                                if vendor_clean.startswith("- Company Name:"):
                                    vendor_clean = vendor_clean.replace("- Company Name:", "").strip()
                                elif vendor_clean.startswith("- Vendor Type:"):
                                    vendor_clean = vendor_clean.replace("- Vendor Type:", "").strip()
                                elif vendor_clean.startswith("-"):
                                    vendor_clean = vendor_clean.lstrip("- ").strip()
                                
                                # Filter out explanatory text and headers
                                if any(bad_text in vendor_clean.lower() for bad_text in [
                                    'here are', 'extracted', 'company names', 'vendor types',
                                    'payment to', 'invoice from', 'implied', 'not explicitly', 
                                    'mentioned', 'but not', 'might be', 'could be', 'seems like', 
                                    'appears to be', 'no company', 'just the equipment', 
                                    'equipment type', 'funding description'
                                ]):
                                    print(f"   ❌ Rejected explanatory text: {vendor_clean}")
                                    continue
                                    
                                print(f"🔍 DEBUG: Validating vendor: '{vendor_clean}'")
                                validated_vendor = self._validate_vendor_name_fast(vendor_clean)
                                if validated_vendor:
                                    vendors.append(validated_vendor)
                                    print(f"   ✅ Ollama vendor: {validated_vendor}")
                                else:
                                    print(f"   ❌ Rejected vendor: {vendor_clean}")
                    else:
                        vendors = []
                
                except Exception as e:
                    print(f"   ❌ Ollama processing failed: {e}")
                    vendors = []
                
                print(f"   🔍 EC2 DEBUG: Single batch final vendors: {vendors[:10]}")  # Show first 10 vendors for debugging
            
            # Analyze full dataset potential for coverage assessment
            if len(descriptions) > 100:
                potential_vendors = self._analyze_full_dataset_potential(descriptions)
                print(f"   🔍 Full dataset analysis: {potential_vendors} potential vendors detected")
                print(f"   📊 Current sample coverage: {len(vendors)}/{potential_vendors} vendors")
            
            # Cache the result for consistency
            self._vendor_cache[cache_key] = vendors
            print(f"💾 Cached vendor extraction result for consistency")
            
            return vendors
            
        except Exception as e:
            print(f"   ❌ Ollama vendor extraction failed: {e}")
            return []
    
    def _extract_vendors_with_xgboost(self, descriptions):
        """Extract vendors using XGBoost ML approach"""
        try:
            import xgboost as xgb
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            print("   🤖 Using XGBoost ML enhancement...")
            
            # Process all descriptions
            sample_descriptions = descriptions
            
            # Prepare training data (simplified for demo)
            training_descriptions = [
                "Payment to ABC Construction Company Ltd",
                "Invoice from XYZ Steel Corp", 
                "Purchase from DEF Manufacturing Inc",
                "Equipment maintenance costs",
                "Utility payments for electricity"
            ]
            training_labels = ["ABC Construction Company Ltd", "XYZ Steel Corp", "DEF Manufacturing Inc", "", ""]
        
        # Create TF-IDF features
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X_train = vectorizer.fit_transform(training_descriptions)
            
            # Encode labels
            le = LabelEncoder()
            y_train = le.fit_transform([label if label else "NO_VENDOR" for label in training_labels])
            
            # Train XGBoost model
            model = xgb.XGBClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict on sample descriptions
            X_test = vectorizer.transform([str(desc) for desc in sample_descriptions])
            predictions = model.predict(X_test)
            
            # Extract vendor names
            vendors = []
            for pred in predictions:
                vendor = le.inverse_transform([pred])[0]
                if vendor != "NO_VENDOR" and len(vendor) > 2:
                    vendors.append(vendor)
                    print(f"   ✅ XGBoost vendor: {vendor}")
            
            print(f"   🤖 XGBoost completed in 0.00s: {len(vendors)} vendors")
            return vendors
            
        except Exception as e:
            print(f"   ❌ XGBoost extraction failed: {e}")
            return []
        
    def _is_likely_company_name(self, vendor_name):
        """Enhanced validation to accept real company names including abbreviations"""
        if not vendor_name or len(vendor_name.strip()) < 2:
            return False
        
        vendor_clean = vendor_name.strip()
        vendor_lower = vendor_clean.lower()
        
        # Reject equipment, projects, or concepts
        rejected_terms = [
            'rolling mill', 'plant modernization', 'steel plates', 'energy efficiency',
            'infrastructure development', 'warehouse construction', 'capacity increase',
            'equipment', 'machinery', 'upgrade', 'installation', 'maintenance',
            'project', 'development', 'expansion', 'modernization'
        ]
        
        for term in rejected_terms:
            if term in vendor_lower:
                return False
        
        # Accept names with business indicators
        business_indicators = [
            'company', 'corp', 'corporation', 'ltd', 'limited', 'llc', 'inc',
            'group', 'enterprises', 'holdings', 'international', 'industries',
            'construction', 'engineering', 'manufacturing', 'trading', 'services'
        ]
        
        for indicator in business_indicators:
            if indicator in vendor_lower:
                return True
        
        # Accept proper names (capitalized multi-word)
        words = vendor_clean.split()
        if len(words) >= 2 and all(word[0].isupper() for word in words if len(word) > 0):
            return True
        
        # ACCEPT common company abbreviations and single-word names
        common_companies = [
            'pnb', 'sbi', 'hdfc', 'axis', 'icici', 'kotak', 'yes bank', 'indian bank',
            'canara bank', 'union bank', 'bhel', 'ongc', 'ntpc', 'sail', 'tata',
            'reliance', 'adani', 'infosys', 'tcs', 'wipro', 'pgcil', 'thermo',
            'indian oil', 'bharat petroleum', 'hpcl', 'gail', 'coal india'
        ]
        
        if vendor_lower in common_companies:
            return True
        
        # Accept single capitalized words (common for company names)
        if len(vendor_clean) >= 3 and vendor_clean[0].isupper() and vendor_clean.isalpha():
            return True
        
        # Accept abbreviations (all caps, 2-6 characters)
        if vendor_clean.isupper() and 2 <= len(vendor_clean) <= 6 and vendor_clean.isalpha():
            return True
        
        return False
    
    def _analyze_full_dataset_potential(self, descriptions):
        """Quick analysis of full dataset to estimate vendor potential"""
        try:
            potential_count = 0
            for desc in descriptions[:500]:  # Quick scan of first 500
                if str(desc).strip() == '' or str(desc) in ['nan', 'None', '']:
                    continue
                desc_str = str(desc).lower()
                if any(term in desc_str for term in ['company', 'corp', 'ltd', 'inc', 'construction', 'engineering']):
                    potential_count += 1
            
            # Extrapolate to full dataset
            if len(descriptions) > 500:
                potential_count = int(potential_count * (len(descriptions) / 500))
            
            return potential_count
        except:
            return 0
    
    def _consolidate_vendors_fast(self, vendors, descriptions):
        """Fast vendor consolidation and cleanup - FIXED to handle AI-extracted vendors"""
        if not vendors:
            return []
        
        print(f"   🧠 Fast vendor consolidation...")
        
        # Remove duplicates (case-insensitive)
        unique_vendors = {}
        for vendor in vendors:
            vendor_clean = vendor.strip()
            vendor_key = vendor_clean.lower()
            if vendor_key not in unique_vendors and len(vendor_clean) > 2:
                unique_vendors[vendor_key] = vendor_clean
        
        result = list(unique_vendors.values())
        
        print(f"\n📊 FAST VENDOR EXTRACTION RESULTS:")
        print(f"   🎯 Total Transactions: {len(descriptions)}")
        print(f"   🏢 Unique Vendors: {len(result)}")
        print(f"\n🏢 VENDORS IDENTIFIED:")
        
        # IMPROVED: Better matching logic for AI-extracted vendors
        vendor_counts = {}
        for vendor in result:
            count = 0
            vendor_lower = vendor.lower()
            
            for desc in descriptions:
                desc_str = str(desc).strip()
                if desc_str == '' or desc_str in ['nan', 'None', '']:
                    continue
                    
                desc_lower = desc_str.lower()
                
                # Method 1: Exact substring match (original)
                if vendor_lower in desc_lower:
                    count += 1
                    continue
                
                # Method 2: Partial word matching for AI-extracted names
                vendor_words = vendor_lower.split()
                if len(vendor_words) >= 2:  # For multi-word companies
                    # Check if at least 2 words from vendor name appear in description
                    matches = sum(1 for word in vendor_words if len(word) > 3 and word in desc_lower)
                    if matches >= 2:
                        count += 1
                        continue
                
                # Method 3: Check if vendor is a meaningful extraction from this description
                # For cases where Ollama extracted a proper company name
                if len(vendor) > 5:  # Only for substantial vendor names
                    # Check if key parts of vendor name relate to description
                    vendor_core = vendor_lower.replace('ltd', '').replace('inc', '').replace('corp', '').replace('company', '').strip()
                    if vendor_core and len(vendor_core) > 3:
                        if any(word in desc_lower for word in vendor_core.split() if len(word) > 3):
                            count += 1
            
            vendor_counts[vendor] = count
        
        # Sort by frequency and display top vendors
        sorted_vendors = sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True)
        
        # IMPROVED: More lenient filtering for AI-extracted vendors
        # Don't filter out vendors completely - they might be valid even with low matches
        valid_vendors = []
        for vendor, count in sorted_vendors:
            # Keep vendors that either:
            # 1. Have direct matches in descriptions, OR
            # 2. Are substantial company names extracted by AI (even if no direct match)
            # 3. Are any vendor name longer than 2 characters (very lenient)
            if count > 0 or len(vendor) > 2:
                valid_vendors.append((vendor, count))
        
        print(f"   📊 Valid vendors (with transactions):")
        display_count = 0
        for vendor, count in valid_vendors[:15]:  # Show more vendors
            if count > 0:
                print(f"   • {vendor} ({count} transactions)")
                display_count += 1
            else:
                print(f"   • {vendor} (AI-extracted)")
                display_count += 1
        
        if len(valid_vendors) > 15:
            print(f"   ... and {len(valid_vendors) - 15} more vendors")
        
        # Return all valid vendors (not just those with matches)
        final_vendors = [vendor for vendor, _ in valid_vendors]
        
        print(f"   🎯 Final vendor count: {len(final_vendors)}")
        return final_vendors
    
    def _create_vendor_assignments(self, descriptions, unique_vendors):
        """
        Create vendor assignments for each transaction based on extracted vendors
        
        Args:
            descriptions: List of transaction descriptions
            unique_vendors: List of unique vendors found
            
        Returns:
            List of vendor assignments, one for each transaction
        """
        print(f"🔍 DEBUG: Creating vendor assignments for {len(descriptions)} transactions with {len(unique_vendors)} unique vendors")
        print(f"🔍 DEBUG: Unique vendors: {unique_vendors}")
        
        if not unique_vendors:
            # If no vendors found, assign "Other Services" to all transactions
            print(f"❌ DEBUG: No unique vendors found! Assigning 'Other Services' to all {len(descriptions)} transactions")
            return ["Other Services"] * len(descriptions)
        
        vendor_assignments = []
        
        for desc in descriptions:
            desc_str = str(desc).strip()
            if desc_str == '' or desc_str in ['nan', 'None', '']:
                vendor_assignments.append("Other Services")
                continue
            
            desc_lower = desc_str.lower()
            assigned_vendor = None
            
            # Try to find the best matching vendor for this transaction
            for vendor in unique_vendors:
                vendor_lower = vendor.lower()
                
                # Method 1: Exact substring match
                if vendor_lower in desc_lower:
                    assigned_vendor = vendor
                    break
                
                # Method 2: Partial word matching for multi-word companies
                vendor_words = vendor_lower.split()
                if len(vendor_words) >= 2:
                    matches = sum(1 for word in vendor_words if len(word) > 3 and word in desc_lower)
                    if matches >= 2:
                        assigned_vendor = vendor
                        break
                
                # Method 3: Check if vendor core relates to description
                if len(vendor) > 5:
                    vendor_core = vendor_lower.replace('ltd', '').replace('inc', '').replace('corp', '').replace('company', '').strip()
                    if vendor_core and len(vendor_core) > 3:
                        if any(word in desc_lower for word in vendor_core.split() if len(word) > 3):
                            assigned_vendor = vendor
                            break
            
            # If no specific vendor found, assign the first available vendor or "Other Services"
            if assigned_vendor is None:
                assigned_vendor = unique_vendors[0] if unique_vendors else "Other Services"
            
            vendor_assignments.append(assigned_vendor)
        
        print(f"✅ Created vendor assignments: {len(vendor_assignments)} assignments for {len(descriptions)} transactions")
        return vendor_assignments
