#!/usr/bin/env python3
"""
Analytics Analysis Tool for Neural OS Multi-GPU System

This script analyzes the structured analytics logs to generate reports and insights.
Usage: python analyze_analytics.py [--since HOURS] [--type TYPE]
"""

import json
import argparse
import glob
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics

class AnalyticsAnalyzer:
    def __init__(self, since_hours=24):
        self.since_timestamp = time.time() - (since_hours * 3600)
        self.data = {
            'gpu_metrics': [],
            'connection_events': [],
            'queue_metrics': [],
            'ip_stats': []
        }
        self.load_data()
    
    def load_data(self):
        """Load all analytics data files"""
        file_types = {
            'gpu_metrics': 'gpu_metrics_*.jsonl',
            'connection_events': 'connection_events_*.jsonl', 
            'queue_metrics': 'queue_metrics_*.jsonl',
            'ip_stats': 'ip_stats_*.jsonl'
        }
        
        for data_type, pattern in file_types.items():
            files = glob.glob(pattern)
            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            try:
                                record = json.loads(line.strip())
                                if record.get('type') != 'metadata' and record.get('timestamp', 0) >= self.since_timestamp:
                                    self.data[data_type].append(record)
                            except json.JSONDecodeError:
                                continue
                except FileNotFoundError:
                    continue
        
        print(f"Loaded data from the last {(time.time() - self.since_timestamp) / 3600:.1f} hours:")
        for data_type, records in self.data.items():
            print(f"  {data_type}: {len(records)} records")
        print()
    
    def analyze_gpu_utilization(self):
        """Analyze GPU utilization patterns"""
        print("🖥️  GPU UTILIZATION ANALYSIS")
        print("=" * 40)
        
        gpu_records = [r for r in self.data['gpu_metrics'] if r.get('type') == 'gpu_status']
        if not gpu_records:
            print("No GPU utilization data found.")
            return
        
        utilizations = [r['utilization_percent'] for r in gpu_records]
        total_gpus = gpu_records[-1].get('total_gpus', 0)
        
        print(f"Total GPUs: {total_gpus}")
        print(f"Average utilization: {statistics.mean(utilizations):.1f}%")
        print(f"Peak utilization: {max(utilizations):.1f}%")
        print(f"Minimum utilization: {min(utilizations):.1f}%")
        print(f"Utilization std dev: {statistics.stdev(utilizations) if len(utilizations) > 1 else 0:.1f}%")
        
        # Utilization distribution
        high_util = sum(1 for u in utilizations if u >= 80)
        med_util = sum(1 for u in utilizations if 40 <= u < 80)
        low_util = sum(1 for u in utilizations if u < 40)
        
        print(f"\nUtilization distribution:")
        print(f"  High (≥80%): {high_util} samples ({high_util/len(utilizations)*100:.1f}%)")
        print(f"  Medium (40-79%): {med_util} samples ({med_util/len(utilizations)*100:.1f}%)")
        print(f"  Low (<40%): {low_util} samples ({low_util/len(utilizations)*100:.1f}%)")
        print()
    
    def analyze_connections(self):
        """Analyze connection patterns"""
        print("🔗 CONNECTION ANALYSIS")
        print("=" * 40)
        
        opens = [r for r in self.data['connection_events'] if r.get('type') == 'connection_open']
        closes = [r for r in self.data['connection_events'] if r.get('type') == 'connection_close']
        
        if not opens and not closes:
            print("No connection data found.")
            return
        
        print(f"Total connections opened: {len(opens)}")
        print(f"Total connections closed: {len(closes)}")
        
        if closes:
            durations = [r['duration'] for r in closes]
            interactions = [r['interactions'] for r in closes]
            reasons = [r['reason'] for r in closes]
            
            print(f"\nSession durations:")
            print(f"  Average: {statistics.mean(durations):.1f}s")
            print(f"  Median: {statistics.median(durations):.1f}s")
            print(f"  Max: {max(durations):.1f}s")
            print(f"  Min: {min(durations):.1f}s")
            
            print(f"\nInteractions per session:")
            print(f"  Average: {statistics.mean(interactions):.1f}")
            print(f"  Median: {statistics.median(interactions):.1f}")
            print(f"  Max: {max(interactions)}")
            
            print(f"\nSession end reasons:")
            reason_counts = Counter(reasons)
            for reason, count in reason_counts.most_common():
                print(f"  {reason}: {count} ({count/len(closes)*100:.1f}%)")
        print()
    
    def analyze_queue_performance(self):
        """Analyze queue performance"""
        print("📝 QUEUE PERFORMANCE ANALYSIS")
        print("=" * 40)
        
        bypasses = [r for r in self.data['queue_metrics'] if r.get('type') == 'queue_bypass']
        waits = [r for r in self.data['queue_metrics'] if r.get('type') == 'queue_wait']
        statuses = [r for r in self.data['queue_metrics'] if r.get('type') == 'queue_status']
        
        total_users = len(bypasses) + len(waits)
        if total_users == 0:
            print("No queue data found.")
            return
        
        print(f"Total users processed: {total_users}")
        print(f"Users bypassed queue: {len(bypasses)} ({len(bypasses)/total_users*100:.1f}%)")
        print(f"Users waited in queue: {len(waits)} ({len(waits)/total_users*100:.1f}%)")
        
        if waits:
            wait_times = [r['wait_time'] for r in waits]
            positions = [r['queue_position'] for r in waits]
            
            print(f"\nWait time statistics:")
            print(f"  Average wait: {statistics.mean(wait_times):.1f}s")
            print(f"  Median wait: {statistics.median(wait_times):.1f}s")
            print(f"  Max wait: {max(wait_times):.1f}s")
            print(f"  Average queue position: {statistics.mean(positions):.1f}")
        
        if statuses:
            queue_sizes = [r['queue_size'] for r in statuses]
            estimated_waits = [r['estimated_wait'] for r in statuses if r['queue_size'] > 0]
            
            print(f"\nQueue size statistics:")
            print(f"  Average queue size: {statistics.mean(queue_sizes):.1f}")
            print(f"  Max queue size: {max(queue_sizes)}")
            
            if estimated_waits:
                print(f"  Average estimated wait: {statistics.mean(estimated_waits):.1f}s")
        print()
    
    def analyze_ip_usage(self):
        """Analyze IP address usage patterns"""
        print("🌍 IP USAGE ANALYSIS") 
        print("=" * 40)
        
        ip_records = self.data['ip_stats']
        if not ip_records:
            print("No IP usage data found.")
            return
        
        # Get latest connection counts per IP
        latest_ip_data = {}
        for record in ip_records:
            if record.get('type') == 'ip_update':
                ip = record['ip_address']
                latest_ip_data[ip] = record['connection_count']
        
        if not latest_ip_data:
            print("No IP connection data found.")
            return
        
        total_connections = sum(latest_ip_data.values())
        unique_ips = len(latest_ip_data)
        
        print(f"Total unique IP addresses: {unique_ips}")
        print(f"Total connections: {total_connections}")
        print(f"Average connections per IP: {total_connections/unique_ips:.1f}")
        
        print(f"\nTop IP addresses by connection count:")
        sorted_ips = sorted(latest_ip_data.items(), key=lambda x: x[1], reverse=True)
        for i, (ip, count) in enumerate(sorted_ips[:10], 1):
            percentage = count / total_connections * 100
            print(f"  {i:2d}. {ip}: {count} connections ({percentage:.1f}%)")
        print()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("📊 SYSTEM SUMMARY REPORT")
        print("=" * 50)
        
        # Time range
        start_time = datetime.fromtimestamp(self.since_timestamp)
        end_time = datetime.now()
        duration_hours = (end_time.timestamp() - self.since_timestamp) / 3600
        
        print(f"Report period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration_hours:.1f} hours")
        print()
        
        self.analyze_gpu_utilization()
        self.analyze_connections()
        self.analyze_queue_performance()
        self.analyze_ip_usage()

def main():
    parser = argparse.ArgumentParser(description='Analyze Neural OS analytics data')
    parser.add_argument('--since', type=float, default=24, 
                       help='Analyze data from the last N hours (default: 24)')
    parser.add_argument('--type', choices=['gpu', 'connections', 'queue', 'ip', 'summary'],
                       default='summary', help='Type of analysis to perform')
    
    args = parser.parse_args()
    
    analyzer = AnalyticsAnalyzer(since_hours=args.since)
    
    if args.type == 'gpu':
        analyzer.analyze_gpu_utilization()
    elif args.type == 'connections':
        analyzer.analyze_connections()
    elif args.type == 'queue':
        analyzer.analyze_queue_performance()
    elif args.type == 'ip':
        analyzer.analyze_ip_usage()
    else:
        analyzer.generate_summary_report()

if __name__ == '__main__':
    main() 