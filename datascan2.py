#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path

class LargeCacheDataExtractor:
    def __init__(self):
        # Large cache sizes for Section 2 (in bytes) - 128KB to 8MB
        # These correspond to: 128KB, 256KB, 512KB, 1MB, 2MB, 4MB, 8MB
        self.large_cache_sizes = [
            131072,   # 128KB
            262144,   # 256KB  
            524288,   # 512KB
            1048576,  # 1MB
            2097152,  # 2MB
            4194304,  # 4MB
            8388608   # 8MB
        ]
        
        # SPEC92 applications for large cache section (Section 2)
        self.spec92_apps = ['nasa7', 'su2', 'swm', 'wave']
        
        # Cache size labels for user-friendly output
        self.cache_size_labels = {
            131072: "128KB",
            262144: "256KB", 
            524288: "512KB",
            1048576: "1MB",
            2097152: "2MB",
            4194304: "4MB",
            8388608: "8MB"
        }
        
    def extract_hit_ratios(self, filename):
        """
        Extract instruction and data cache hit ratios from an ACS output file.
        
        Args:
            filename (str): Path to the ACS output .txt file
            
        Returns:
            tuple: (instruction_hit_ratio, data_hit_ratio) or (None, None) if error
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Regex patterns to find hit ratios in the L1 I Cache and L1 D Cache sections
            # Looking for patterns like "Hit ratio : 0.908862"
            inst_pattern = r'L1 I Cache.*?Hit ratio\s*:\s*([0-9]+\.?[0-9]*)'
            data_pattern = r'L1 D Cache.*?Hit ratio\s*:\s*([0-9]+\.?[0-9]*)'
            
            # Search for patterns using DOTALL flag to match across newlines
            inst_match = re.search(inst_pattern, content, re.DOTALL | re.IGNORECASE)
            data_match = re.search(data_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if inst_match and data_match:
                inst_hit_ratio = float(inst_match.group(1))
                data_hit_ratio = float(data_match.group(1))
                return inst_hit_ratio, data_hit_ratio
            else:
                print(f"    Warning: Could not find hit ratios in {filename}")
                if not inst_match:
                    print("      - Instruction cache hit ratio not found")
                if not data_match:
                    print("      - Data cache hit ratio not found")
                return None, None
                
        except FileNotFoundError:
            print(f"    Error: File {filename} not found")
            return None, None
        except ValueError as e:
            print(f"    Error: Invalid number format in {filename}: {e}")
            return None, None
        except Exception as e:
            print(f"    Error processing {filename}: {e}")
            return None, None
    
    def read_existing_dat_file(self, filename):
        """
        Read existing .dat file and return list of (cache_size, current_value) tuples.
        
        Args:
            filename (str): Path to the .dat file
            
        Returns:
            list: List of (cache_size, current_value) tuples
        """
        data = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                cache_size = int(parts[0])
                                current_value = parts[1]
                                data.append((cache_size, current_value))
                            except ValueError:
                                print(f"    Warning: Invalid data format in {filename} line {line_num}: {line}")
            return data
        except FileNotFoundError:
            print(f"    Warning: {filename} not found - cannot update")
            return []
        except Exception as e:
            print(f"    Error reading {filename}: {e}")
            return []
    
    def update_dat_file(self, filename, hit_ratios_dict):
        """
        Update existing .dat file by replacing /0 entries with actual hit ratios.
        
        Args:
            filename (str): Path to the .dat file to update
            hit_ratios_dict (dict): Dictionary mapping cache_size to hit_ratio
        """
        try:
            existing_data = self.read_existing_dat_file(filename)
            if not existing_data:
                print(f"    No existing data found in {filename} - skipping")
                return
                
            updated_lines = []
            updated_count = 0
            
            for cache_size, current_value in existing_data:
                if current_value == '/0' and cache_size in hit_ratios_dict:
                    new_value = hit_ratios_dict[cache_size]
                    if new_value is not None:
                        updated_lines.append(f"{cache_size}\t{new_value:.6f}")
                        updated_count += 1
                        size_label = self.cache_size_labels.get(cache_size, f"{cache_size}B")
                        print(f"      ✓ Updated {size_label}: /0 -> {new_value:.6f}")
                    else:
                        updated_lines.append(f"{cache_size}\t/0")
                        size_label = self.cache_size_labels.get(cache_size, f"{cache_size}B")
                        print(f"      - Kept {size_label}: /0 (no data available)")
                else:
                    updated_lines.append(f"{cache_size}\t{current_value}")
            
            # Write back to file with backup
            backup_filename = f"{filename}.backup"
            if os.path.exists(filename):
                os.rename(filename, backup_filename)
                
            with open(filename, 'w', encoding='utf-8') as f:
                for line in updated_lines:
                    f.write(line + '\n')
                    
            print(f"    ✓ Successfully updated {filename}: {updated_count} entries filled")
            
            # Remove backup if successful
            if os.path.exists(backup_filename):
                os.remove(backup_filename)
                
        except Exception as e:
            print(f"    Error updating {filename}: {e}")
            # Restore backup if it exists
            backup_filename = f"{filename}.backup"
            if os.path.exists(backup_filename):
                os.rename(backup_filename, filename)
                print(f"    Restored backup for {filename}")
    
    def fill_application_data(self, app_name):
        """
        Fill existing .dat files for a given application by extracting data from large cache .txt files.
        
        Args:
            app_name (str): Name of the SPEC92 application (e.g., 'nasa7', 'su2', 'swm', 'wave')
        """
        inst_ratios = {}
        data_ratios = {}
        
        print(f"\n🔍 Processing {app_name.upper()} - Large Cache Data (Section 2):")
        print("=" * 60)
        
        files_found = 0
        files_processed = 0
        
        for size in self.large_cache_sizes:
            filename = f"{app_name}-{size}.txt"
            size_label = self.cache_size_labels[size]
            
            if os.path.exists(filename):
                files_found += 1
                print(f"  📁 Found {filename} ({size_label})")
                
                inst_ratio, data_ratio = self.extract_hit_ratios(filename)
                if inst_ratio is not None and data_ratio is not None:
                    inst_ratios[size] = inst_ratio
                    data_ratios[size] = data_ratio
                    files_processed += 1
                    print(f"    ✓ Extracted: I={inst_ratio:.6f}, D={data_ratio:.6f}")
                else:
                    print(f"    ✗ Failed to extract hit ratios")
            else:
                print(f"  - Missing {filename} ({size_label})")
        
        print(f"\n📊 Summary for {app_name}:")
        print(f"  Files found: {files_found}/{len(self.large_cache_sizes)}")
        print(f"  Files processed: {files_processed}/{files_found}")
        
        if files_processed == 0:
            print(f"  ⚠️  No .txt files could be processed for {app_name}")
            return
            
        # Update existing .dat files
        inst_file = f"instruction-{app_name}.dat"
        data_file = f"data-{app_name}.dat"
        
        print(f"\n📝 Updating .dat files for {app_name}:")
        
        if os.path.exists(inst_file) and inst_ratios:
            print(f"  🔄 Updating {inst_file}...")
            self.update_dat_file(inst_file, inst_ratios)
        elif not os.path.exists(inst_file):
            print(f"  ⚠️  Warning: {inst_file} does not exist")
        elif not inst_ratios:
            print(f"  ℹ️  No instruction cache data to update {inst_file}")
            
        if os.path.exists(data_file) and data_ratios:
            print(f"  🔄 Updating {data_file}...")
            self.update_dat_file(data_file, data_ratios)
        elif not os.path.exists(data_file):
            print(f"  ⚠️  Warning: {data_file} does not exist")
        elif not data_ratios:
            print(f"  ℹ️  No data cache data to update {data_file}")
    
    def test_large_cache_files(self):
        """Test the extractor with any available large cache files."""
        print("=" * 70)
        print("🧪 TESTING LARGE CACHE DATA EXTRACTOR")
        print("=" * 70)
        
        print("Looking for large cache test files...")
        
        total_files_found = 0
        for app in self.spec92_apps:
            app_files = []
            for size in self.large_cache_sizes:
                filename = f"{app}-{size}.txt"
                if os.path.exists(filename):
                    app_files.append((filename, self.cache_size_labels[size]))
                    total_files_found += 1
            
            if app_files:
                print(f"\n📱 {app.upper()} files found:")
                for filename, size_label in app_files:
                    inst_ratio, data_ratio = self.extract_hit_ratios(filename)
                    print(f"  {filename} ({size_label}):")
                    print(f"    Instruction Hit Ratio: {inst_ratio}")
                    print(f"    Data Hit Ratio: {data_ratio}")
        
        if total_files_found == 0:
            print("\n❌ No large cache .txt files found in current directory.")
            print("Expected files like: nasa7-131072.txt, su2-1048576.txt, etc.")
            print("Make sure you've run the spec-dm-big-run script first.")
        else:
            print(f"\n✅ Found {total_files_found} large cache test files total.")
    
    def fill_all_large_cache_data(self):
        """Fill all SPEC92 large cache .dat files."""
        print("=" * 70)
        print("🚀 LARGE CACHE HIT RATIO EXTRACTOR (SECTION 2)")
        print("=" * 70)
        print("Processing SPEC92 Applications - Large Cache Data (128KB - 8MB)")
        print("Applications: nasa7, su2, swm, wave")
        print("=" * 70)
        
        total_apps_processed = 0
        
        for app in self.spec92_apps:
            self.fill_application_data(app)
            total_apps_processed += 1
        
        print("=" * 70)
        print("🎉 LARGE CACHE PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"✅ Processed {total_apps_processed} SPEC92 applications")
        print("\n📁 Updated .dat files:")
        for app in self.spec92_apps:
            print(f"  - instruction-{app}.dat")
            print(f"  - data-{app}.dat")
        
        print("\n🎯 Next steps:")
        print("  1. Use gnuplot to generate large cache performance plots:")
        print("     gnuplot spec-cache-big-r2000.plt")
        print("  2. View the generated plots: spec92-dm-big.ps or spec92-dm-big.png")
        print("  3. Analyze the results for Task 3 questions")

def print_usage():
    """Print usage information."""
    print("FIT3159 Lab 6 - Large Cache Hit Ratio Extractor (Section 2)")
    print("=" * 65)
    print("Usage: python3 large_cache_extractor.py [command]")
    print()
    print("Commands:")
    print("  test        - Test with any available large cache files")
    print("  nasa7       - Process nasa7 large cache data only")
    print("  su2         - Process su2 large cache data only")  
    print("  swm         - Process swm large cache data only")
    print("  wave        - Process wave large cache data only")
    print("  all         - Process all SPEC92 large cache data")
    print("  help        - Show this help message")
    print()
    print("If no command is provided, processes all large cache data.")
    print()
    print("Cache sizes processed: 128KB, 256KB, 512KB, 1MB, 2MB, 4MB, 8MB")
    print()
    print("Required files:")
    print("  Input:  [app]-[size].txt (e.g., nasa7-131072.txt, su2-1048576.txt)")
    print("  Output: instruction-[app].dat and data-[app].dat")
    print()
    print("Note: Run spec-dm-big-run script first to generate input .txt files")

def main():
    """Main function to run the large cache data extractor."""
    extractor = LargeCacheDataExtractor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['help', '-h', '--help']:
            print_usage()
            return
        elif command == 'test':
            extractor.test_large_cache_files()
        elif command in ['nasa7', 'su2', 'swm', 'wave']:
            if command in extractor.spec92_apps:
                print(f"Processing {command.upper()} large cache data only...")
                extractor.fill_application_data(command)
            else:
                print(f"Error: {command} is not a valid SPEC92 application")
                print("Valid applications: nasa7, su2, swm, wave")
        elif command == 'all':
            extractor.fill_all_large_cache_data()
        else:
            print(f"Unknown command: {command}")
            print_usage()
            return
    else:
        # Default behavior: fill all large cache data
        print("Large Cache Hit Ratio Extractor - FIT3159 Lab 6 Section 2")
        print("=" * 65)
        print("No command specified - processing all SPEC92 large cache data...")
        print()
        
        extractor.fill_all_large_cache_data()

if __name__ == "__main__":
    main()