
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class VehicleAnalytics:
    
    def __init__(self, detections):
        self.detections = detections
        self.parsed_data = self._parse_detections()
    
    def _parse_detections(self):
        parsed = []
        for det in self.detections:
            try:
                # format: (id, total_spots, occupied_spots, available_spots, occupancy_rate, timestamp)
                # Handle both PostgreSQL datetime and CSV string
                timestamp_idx = 5 if len(det) > 5 else 2
                
                if isinstance(det[timestamp_idx], str):
                    timestamp = datetime.strptime(det[timestamp_idx], '%Y-%m-%d %H:%M:%S')
                else:
                    timestamp = det[timestamp_idx]
                
                parsed.append({
                    'id': det[0],
                    'total': int(det[1]) if len(det) > 5 else 0,
                    'count': int(det[2]) if len(det) > 5 else int(det[1]),
                    'available': int(det[3]) if len(det) > 5 else 0,
                    'occupancy_rate': float(det[4]) if len(det) > 5 else 0.0,
                    'timestamp': timestamp
                })
            except Exception as e:
                print(f"Error parsing detection: {e}")
                continue
        
        return sorted(parsed, key=lambda x: x['timestamp'])
    
    def get_hourly_distribution(self):
        hourly_counts = defaultdict(list)
        
        for det in self.parsed_data:
            hour = det['timestamp'].hour
            hourly_counts[hour].append(det['count'])
        
        hourly_stats = {}
        for hour in range(24):
            if hour in hourly_counts:
                counts = hourly_counts[hour]
                hourly_stats[hour] = {
                    'hour': hour,
                    'average': round(statistics.mean(counts), 2),
                    'max': max(counts),
                    'min': min(counts),
                    'samples': len(counts)
                }
            else:
                hourly_stats[hour] = {
                    'hour': hour,
                    'average': 0,
                    'max': 0,
                    'min': 0,
                    'samples': 0
                }
        
        return hourly_stats
    
    def get_peak_hours(self, top_n=5):
        hourly = self.get_hourly_distribution()
        sorted_hours = sorted(
            hourly.values(), 
            key=lambda x: x['average'], 
            reverse=True
        )
        return sorted_hours[:top_n]
    
    def get_daily_summary(self):
        daily_counts = defaultdict(list)
        
        for det in self.parsed_data:
            day = det['timestamp'].date()
            daily_counts[day].append(det['count'])
        
        daily_stats = []
        for day, counts in sorted(daily_counts.items()):
            daily_stats.append({
                'date': str(day),
                'average': round(statistics.mean(counts), 2),
                'max': max(counts),
                'min': min(counts),
                'total_samples': len(counts)
            })
        
        return daily_stats
    
    def get_time_range_stats(self, start_time=None, end_time=None):
        if not start_time and self.parsed_data:
            start_time = self.parsed_data[0]['timestamp']
        if not end_time:
            end_time = datetime.now()
        
        filtered = [
            det for det in self.parsed_data
            if start_time <= det['timestamp'] <= end_time
        ]
        
        if not filtered:
            return {
                'start_time': str(start_time),
                'end_time': str(end_time),
                'total_records': 0,
                'average': 0,
                'max': 0,
                'min': 0
            }
        
        counts = [det['count'] for det in filtered]
        
        return {
            'start_time': str(start_time),
            'end_time': str(end_time),
            'total_records': len(filtered),
            'average': round(statistics.mean(counts), 2),
            'max': max(counts),
            'min': min(counts),
            'median': round(statistics.median(counts), 2),
            'std_dev': round(statistics.stdev(counts), 2) if len(counts) > 1 else 0
        }
    
    def get_trend_analysis(self):
        if len(self.parsed_data) < 2:
            return {'trend': 'insufficient_data'}
        
        # Compare first half vs second half average
        mid = len(self.parsed_data) // 2
        first_half = [det['count'] for det in self.parsed_data[:mid]]
        second_half = [det['count'] for det in self.parsed_data[mid:]]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        change_percent = ((avg_second - avg_first) / avg_first * 100) if avg_first > 0 else 0
        
        if change_percent > 10:
            trend = 'increasing'
        elif change_percent < -10:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'first_half_avg': round(avg_first, 2),
            'second_half_avg': round(avg_second, 2),
            'change_percent': round(change_percent, 2)
        }
    
    def get_occupancy_levels(self):
        if not self.parsed_data:
            return {}
        
        counts = [det['count'] for det in self.parsed_data]
        max_count = max(counts)
        
        # Define thresholds
        low_threshold = max_count * 0.3
        medium_threshold = max_count * 0.6
        
        levels = {'low': 0, 'medium': 0, 'high': 0}
        
        for count in counts:
            if count <= low_threshold:
                levels['low'] += 1
            elif count <= medium_threshold:
                levels['medium'] += 1
            else:
                levels['high'] += 1
        
        total = len(counts)
        
        return {
            'low': {
                'count': levels['low'],
                'percentage': round(levels['low'] / total * 100, 2)
            },
            'medium': {
                'count': levels['medium'],
                'percentage': round(levels['medium'] / total * 100, 2)
            },
            'high': {
                'count': levels['high'],
                'percentage': round(levels['high'] / total * 100, 2)
            }
        }
    
    def get_comprehensive_report(self):
        if not self.parsed_data:
            return {'error': 'No data available for analysis'}
        
        return {
            'overview': {
                'total_records': len(self.parsed_data),
                'time_span': {
                    'start': str(self.parsed_data[0]['timestamp']),
                    'end': str(self.parsed_data[-1]['timestamp'])
                },
                'overall_stats': {
                    'average': round(statistics.mean([d['count'] for d in self.parsed_data]), 2),
                    'max': max([d['count'] for d in self.parsed_data]),
                    'min': min([d['count'] for d in self.parsed_data])
                }
            },
            'peak_hours': self.get_peak_hours(5),
            'daily_summary': self.get_daily_summary(),
            'trend': self.get_trend_analysis(),
            'occupancy_levels': self.get_occupancy_levels()
        }

