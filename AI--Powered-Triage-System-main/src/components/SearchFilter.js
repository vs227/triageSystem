import React, { useState } from 'react';
import './SearchFilter.css';

const SearchFilter = ({ onSearch, onClear }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [triageFilter, setTriageFilter] = useState('');
  const [ageRange, setAgeRange] = useState({ min: '', max: '' });

  const handleSearch = () => {
    const filters = {};
    if (searchQuery.trim()) filters.q = searchQuery.trim();
    if (triageFilter) filters.triage_level = triageFilter;
    if (ageRange.min && ageRange.min !== '') {
      const minAge = parseInt(ageRange.min);
      if (!isNaN(minAge)) filters.min_age = minAge;
    }
    if (ageRange.max && ageRange.max !== '') {
      const maxAge = parseInt(ageRange.max);
      if (!isNaN(maxAge)) filters.max_age = maxAge;
    }
    
    if (Object.keys(filters).length > 0) {
      onSearch(filters);
    } else {
      alert('Please enter at least one search criteria:\n• Search text (complaint, gender, history)\n• Triage level\n• Age range (min and/or max)');
    }
  };

  const handleClear = () => {
    setSearchQuery('');
    setTriageFilter('');
    setAgeRange({ min: '', max: '' });
    onClear();
  };

  return (
    <div className="search-filter">
      <h3>🔍 Search & Filter</h3>
      <div className="search-controls">
        <input
          type="text"
          placeholder="Search by complaint, gender, history..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="search-input"
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
        />
        
        <select
          value={triageFilter}
          onChange={(e) => setTriageFilter(e.target.value)}
          className="filter-select"
        >
          <option value="">All Triage Levels</option>
          <option value="Critical">Critical</option>
          <option value="Moderate">Moderate</option>
          <option value="Low">Low</option>
        </select>

        <div className="age-range">
          <input
            type="number"
            placeholder="Min Age"
            value={ageRange.min}
            onChange={(e) => setAgeRange({ ...ageRange, min: e.target.value })}
            className="age-input"
            min="0"
            max="120"
          />
          <span>-</span>
          <input
            type="number"
            placeholder="Max Age"
            value={ageRange.max}
            onChange={(e) => setAgeRange({ ...ageRange, max: e.target.value })}
            className="age-input"
            min="0"
            max="120"
          />
        </div>

        <div className="search-buttons">
          <button onClick={handleSearch} className="search-btn">
            Search
          </button>
          <button onClick={handleClear} className="clear-btn">
            Clear
          </button>
        </div>
      </div>
    </div>
  );
};

export default SearchFilter;
