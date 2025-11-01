import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './Stats.css';

const Stats = ({ stats }) => {
  const chartData = Object.entries(stats.triage_distribution || {}).map(([level, count]) => {
    let label = level;
    if (level === '1' || level === 1) label = 'Critical';
    else if (level === '2' || level === 2) label = 'Moderate';
    else if (level === '3' || level === 3) label = 'Low';
    else if (typeof level === 'string') {
      label = level;
    }
    
    return {
      level: label,
      count: count
    };
  });

  if (chartData.length === 0) {
    chartData.push({ level: 'No Data', count: 0 });
  }

  return (
    <div className="stats">
      <h2>Statistics</h2>
      <div className="stats-content">
        <div className="total-patients">
          <span className="stat-label">Total Patients in Queue:</span>
          <span className="stat-value">{stats.total_patients || 0}</span>
        </div>
        
        <div className="distribution-chart">
          <h3>Triage Level Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#4A90E2" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="triage-breakdown">
          <h3>Breakdown</h3>
          <div className="breakdown-list">
            {Object.entries(stats.triage_distribution || {}).map(([level, count]) => {
              let displayLabel = level;
              let priority = 3;
              const levelStr = String(level).toLowerCase();
              
              if (levelStr.includes('critical') || level === '1' || level === 1) {
                displayLabel = 'Critical';
                priority = 1;
              } else if (levelStr.includes('moderate') || level === '2' || level === 2) {
                displayLabel = 'Moderate';
                priority = 2;
              } else if (levelStr.includes('low') || level === '3' || level === 3) {
                displayLabel = 'Low';
                priority = 3;
              }
              
              return (
                <div key={level} className="breakdown-item">
                  <span className={`triage-indicator level-${priority}`}>
                    {displayLabel}
                  </span>
                  <span className="breakdown-count">{count}</span>
                </div>
              );
            })}
            {Object.keys(stats.triage_distribution || {}).length === 0 && (
              <p className="no-data">No patients in queue</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Stats;
