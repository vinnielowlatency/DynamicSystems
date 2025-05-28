#!/usr/bin/env python3


import os
import re
import sys
from pathlib import Path

class CacheDataExtractor:
    def __init__(self):
        # Cache sizes for small caches (Section 1) - up to 64KB
        self.small_cache_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        
        # Cache sizes for large caches (Section 2) - up to 8MB
        self.large_cache_sizes = [131072, 262144, 524288, 1048576, 
                                 2097152, 4194304, 8388608]
        
        # Application lists
        self.unix_apps = ['awk', 'sed', 'tex', 'yacc']
        self.spec92_apps = ['nasa7', 'su2', 'swm', 'wave']
        
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
                print(f"Warning: Could not find hit ratios in {filename}")
                if not inst_match:
                    print("  - Instruction cache hit ratio not found")
                if not data_match:
                    print("  - Data cache hit ratio not found")
                return None, None
                
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            return None, None
        except ValueError as e:
            print(f"Error: Invalid number format in {filename}: {e}")
            return None, None
        except Exception as e:
            print(f"Error processing {filename}: {e}")
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
                                print(f"Warning: Invalid data format in {filename} line {line_num}: {line}")
            return data
        except FileNotFoundError:
            print(f"Warning: {filename} not found - cannot update")
            return []
        except Exception as e:
            print(f"Error reading {filename}: {e}")
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
                print(f"No existing data found in {filename} - skipping")
                return
                
            updated_lines = []
            updated_count = 0
            
            for cache_size, current_value in existing_data:
                if current_value == '/0' and cache_size in hit_ratios_dict:
                    new_value = hit_ratios_dict[cache_size]
                    if new_value is not None:
                        updated_lines.append(f"{cache_size}\t{new_value:.6f}")
                        updated_count += 1
                        print(f"  Updated {cache_size}: /0 -> {new_value:.6f}")
                    else:
                        updated_lines.append(f"{cache_size}\t/0")
                        print(f"  Kept {cache_size}: /0 (no data available)")
                else:
                    updated_lines.append(f"{cache_size}\t{current_value}")
            
            # Write back to file with backup
            backup_filename = f"{filename}.backup"
            if os.path.exists(filename):
                os.rename(filename, backup_filename)
                
            with open(filename, 'w', encoding='utf-8') as f:
                for line in updated_lines:
                    f.write(line + '\n')
                    
            print(f"Successfully updated {filename}: {updated_count} entries filled")
            
            # Remove backup if successful
            if os.path.exists(backup_filename):
                os.remove(backup_filename)
                
        except Exception as e:
            print(f"Error updating {filename}: {e}")
            # Restore backup if it exists
            backup_filename = f"{filename}.backup"
            if os.path.exists(backup_filename):
                os.rename(backup_filename, filename)
                print(f"Restored backup for {filename}")
    
    def fill_application_data(self, app_name, cache_sizes=None, section_name=""):
        """
        Fill existing .dat files for a given application by extracting data from .txt files.
        
        Args:
            app_name (str): Name of the application (e.g., 'awk', 'nasa7')
            cache_sizes (list): List of cache sizes to process
            section_name (str): Description for logging
        """
        if cache_sizes is None:
            cache_sizes = self.small_cache_sizes
            
        inst_ratios = {}
        data_ratios = {}
        
        print(f"\nProcessing {app_name} {section_name}:")
        print("-" * 50)
        
        files_found = 0
        for size in cache_sizes:
            filename = f"{app_name}-{size}.txt"
            
            if os.path.exists(filename):
                files_found += 1
                inst_ratio, data_ratio = self.extract_hit_ratios(filename)
                if inst_ratio is not None and data_ratio is not None:
                    inst_ratios[size] = inst_ratio
                    data_ratios[size] = data_ratio
                    print(f"  ✓ {filename}: I={inst_ratio:.6f}, D={data_ratio:.6f}")
                else:
                    print(f"  ✗ {filename}: Failed to extract hit ratios")
            else:
                print(f"  - {filename}: File not found")
        
        if files_found == 0:
            print(f"  No .txt files found for {app_name}")
            return
            
        # Update existing .dat files
        inst_file = f"instruction-{app_name}.dat"
        data_file = f"data-{app_name}.dat"
        
        print(f"\nUpdating .dat files for {app_name}:")
        
        if os.path.exists(inst_file) and inst_ratios:
            print(f"  Updating {inst_file}...")
            self.update_dat_file(inst_file, inst_ratios)
        elif not os.path.exists(inst_file):
            print(f"  Warning: {inst_file} does not exist")
        elif not inst_ratios:
            print(f"  No instruction cache data to update {inst_file}")
            
        if os.path.exists(data_file) and data_ratios:
            print(f"  Updating {data_file}...")
            self.update_dat_file(data_file, data_ratios)
        elif not os.path.exists(data_file):
            print(f"  Warning: {data_file} does not exist")
        elif not data_ratios:
            print(f"  No data cache data to update {data_file}")
    
    def test_with_nasa7(self):
        """Test the extractor with the provided nasa7 sample files."""
        print("=" * 60)
        print("TESTING WITH PROVIDED NASA7 SAMPLE FILES")
        print("=" * 60)
        
        # Test files that should exist for nasa7
        test_files = []
        for size in self.small_cache_sizes:
            test_files.append(f"nasa7-{size}.txt")
            
        print("Looking for nasa7 test files...")
        files_found = []
        for filename in test_files:
            if os.path.exists(filename):
                files_found.append(filename)
                
        if not files_found:
            print("No nasa7-*.txt files found in current directory.")
            print("Expected files like: nasa7-1024.txt, nasa7-2048.txt, etc.")
            return
            
        print(f"Found {len(files_found)} nasa7 test files:")
        for filename in files_found:
            inst_ratio, data_ratio = self.extract_hit_ratios(filename)
            print(f"  {filename}:")
            print(f"    Instruction Hit Ratio: {inst_ratio}")
            print(f"    Data Hit Ratio: {data_ratio}")
            
        # Test updating the actual nasa7 .dat files
        self.fill_application_data('nasa7', self.small_cache_sizes, "(Test)")
    
    def fill_unix_utilities(self):
        """Fill existing .dat files for Unix utility applications (Section 1)."""
        print("=" * 60)
        print("SECTION 1: UNIX UTILITIES - SMALL CACHES")
        print("=" * 60)
        
        for app in self.unix_apps:
            self.fill_application_data(app, self.small_cache_sizes, "(Unix Utility)")
    
    def fill_spec92_small(self):
        """Fill existing .dat files for SPEC92 applications with small caches (Section 1)."""
        print("=" * 60)
        print("SECTION 1: SPEC92 BENCHMARKS - SMALL CACHES")
        print("=" * 60)
        
        for app in self.spec92_apps:
            self.fill_application_data(app, self.small_cache_sizes, "(SPEC92 Small)")
    
    def fill_spec92_large(self):
        """Fill existing .dat files for SPEC92 applications with large caches (Section 2)."""
        print("=" * 60)
        print("SECTION 2: SPEC92 BENCHMARKS - LARGE CACHES")
        print("=" * 60)
        
        for app in self.spec92_apps:
            self.fill_application_data(app, self.large_cache_sizes, "(SPEC92 Large)")

def print_usage():
    """Print usage information."""
    print("FIT3159 Lab 6 Task 1 - Cache Hit Ratio Extractor")
    print("=" * 50)
    print("Usage: python3 cache_extractor.py [command]")
    print()
    print("Commands:")
    print("  test        - Test with provided nasa7 sample files")
    print("  unix        - Process Unix utilities (awk, sed, tex, yacc)")
    print("  spec92      - Process SPEC92 benchmarks (nasa7, su2, swm, wave)")
    print("  large       - Process large cache SPEC92 data (Section 2)")
    print("  all         - Process all small cache data (Section 1)")
    print("  help        - Show this help message")
    print()
    print("If no command is provided, processes all Section 1 data.")
    print()
    print("Required files:")
    print("  Input:  [app]-[size].txt (e.g., awk-1024.txt, nasa7-8192.txt)")
    print("  Output: instruction-[app].dat and data-[app].dat")

def main():
    """Main function to run the cache data extractor."""
    extractor = CacheDataExtractor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['help', '-h', '--help']:
            print_usage()
            return
        elif command == 'test':
            extractor.test_with_nasa7()
        elif command == 'unix':
            extractor.fill_unix_utilities()
        elif command == 'spec92':
            extractor.fill_spec92_small()
        elif command == 'large':
            extractor.fill_spec92_large()
        elif command == 'all':
            print("Processing all Section 1 data (Unix utilities + SPEC92 small caches)...")
            extractor.fill_unix_utilities()
            extractor.fill_spec92_small()
        else:
            print(f"Unknown command: {command}")
            print_usage()
            return
    else:
        # Default behavior: fill all Section 1 data
        print("Cache Hit Ratio Extractor - FIT3159 Lab 6 Task 1")
        print("=" * 60)
        print("No command specified - processing all Section 1 data...")
        print("This includes Unix utilities and SPEC92 small cache benchmarks.")
        print()
        
        extractor.fill_unix_utilities()
        extractor.fill_spec92_small()
        
        print("=" * 60)
        print("✓ COMPLETED: All Section 1 .dat files have been updated.")
        print()
        print("Updated applications:")
        print("  Unix utilities: awk, sed, tex, yacc")
        print("  SPEC92 benchmarks: nasa7, su2, swm, wave")
        print()
        print("You can now use gnuplot to generate performance plots:")
        print("  gnuplot util-cache-r2000.plt")
        print("  gnuplot spec-cache-dm-r2000.plt")

if __name__ == "__main__":
    main()