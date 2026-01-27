import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Modal from './Modal';

const F1 = () => {
  const [year, setYear] = useState(new Date().getFullYear());
  const [calendar, setCalendar] = useState([]);
  const [driversStandings, setDriversStandings] = useState([]);
  const [constructorsStandings, setConstructorsStandings] = useState([]);
  const [qualiPredictions, setQualiPredictions] = useState([]);
  const [racePredictions, setRacePredictions] = useState([]);
  const [pointsHistory, setPointsHistory] = useState({ drivers: [], teams: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('Calendar');
  const [nextRace, setNextRace] = useState(null);
  const [predictionsLoaded, setPredictionsLoaded] = useState(false);
  const [raceResults, setRaceResults] = useState([]);
  const [qualiResults, setQualiResults] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState('');
  const [modalContent, setModalContent] = useState(null);

  window.f1PredictionsTab = () => setActiveTab('Predictions');

  useEffect(() => {
    setLoading(true);
    Promise.all([
      axios.get(`http://localhost:8003/api/results/race/${year}`),
      axios.get(`http://localhost:8003/api/results/qualifying/${year}`)
    ])
      .then(([raceResultsResponse, qualiResultsResponse]) => {
        setRaceResults(raceResultsResponse.data);
        setQualiResults(qualiResultsResponse.data);
        setLoading(false);
      })
      .catch(error => {
        setError('Failed to load driver results. Please try again.');
        setLoading(false);
        console.error('Error fetching driver results:', error);
      });
  }, [year]);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      axios.get(`http://localhost:8003/api/calendar/${year}`),
      axios.get(`http://localhost:8003/api/standings/drivers/${year}`),
      axios.get(`http://localhost:8003/api/standings/constructors/${year}`),
      axios.get(`http://localhost:8002/api/next_grand_prix`)
    ])
      .then(([calendarResponse, driversResponse, constructorsResponse, nextRaceResponse]) => {
        setCalendar(calendarResponse.data);
        setDriversStandings(driversResponse.data);
        setConstructorsStandings(constructorsResponse.data);
        setNextRace(nextRaceResponse.data);
        setLoading(false);
      })
      .catch(error => {
        setError('Failed to load data. Please try again.');
        setLoading(false);
        console.error('Error fetching data:', error);
      });
  }, [year]);

  window.f1StatsTab = () => setActiveTab('Season Statistics');

  const loadPredictions = () => {
    setLoading(true);
    Promise.all([
      axios.get('http://localhost:8002/api/predictions/quali'),
      axios.get('http://localhost:8002/api/predictions/race')
    ])
      .then(([qualiResponse, raceResponse]) => {
        setQualiPredictions(qualiResponse.data);
        setRacePredictions(raceResponse.data);
        setPredictionsLoaded(true);
        setLoading(false);
      })
      .catch(error => {
        setError('Failed to load predictions. Please try again.');
        setLoading(false);
        console.error('Error fetching predictions:', error);
      });
  };

  const handleGrandPrixClick = async (year, grandPrix) => {
    try {
      const response = await axios.get(`http://localhost:8003/api/race_results/${year}/${grandPrix}`);
      const data = response.data;
      if (data.length > 0) {
        setModalTitle(`Race Results - ${grandPrix} ${year}`);
        setModalContent(
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white rounded-lg shadow-md">
              <thead className="bg-gray-200">
                <tr>
                  <th className="px-4 py-2 text-left">Driver</th>
                  <th className="px-4 py-2 text-left">Finish Position</th>
                  <th className="px-4 py-2 text-left">Grid Position</th>
                </tr>
              </thead>
              <tbody>
                {data.map((result, index) => (
                  <tr key={index} className="border-t">
                    <td className="px-4 py-2">{result.Driver}</td>
                    <td className="px-4 py-2">{result.FinishPosition}</td> 
                    <td className="px-4 py-2">{result.GridPosition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        setModalOpen(true);
      } else {
        alert('No data available for this race.');
      }
    } catch (error) {
      console.error('Error fetching race results:', error);
      alert('Failed to fetch race results.');
    }
  };

  const handleDriverClick = async (driverId) => {
    try {
      const response = await axios.get(`http://localhost:8003/api/driver_stats/${driverId}`);
      const data = response.data;
      setModalTitle(`Driver Stats - ${data.driver}`);
      setModalContent(
        <div>
          <p><strong>Total Entries:</strong>  {data.total_entries}</p>
          <p><strong>First Race:</strong> {data.first_race.year} - {data.first_race.grand_prix}</p>
          <p><strong>Last Race:</strong> {data.last_race.year} - {data.last_race.grand_prix}</p>
          <p><strong>Best Grid Position:</strong> {data.best_grid_position} ({data.best_grid_count} times)</p>
          <p><strong>Best Race Result:</strong> {data.best_race_result} ({data.best_race_count} times)</p>
          <p><strong>Best Championship Position:</strong> {data.best_championship_position} (Years: {data.best_championship_years})</p>
          <p><strong>Race Wins:</strong> {data.race_wins}</p>
          <p><strong>Pole Positions:</strong> {data.pole_positions}</p>
          <p><strong>Podiums:</strong> {data.podiums}</p>
          <p><strong>First Podium:</strong> {data.first_podium.year} - {data.first_podium.grand_prix}</p>
          <p><strong>Last Podium:</strong> {data.last_podium.year} - {data.last_podium.grand_prix}</p>
          <p><strong>First Win:</strong> {data.first_win.year} - {data.first_win.grand_prix}</p>
          <p><strong>Last Win:</strong> {data.last_win.year} - {data.last_win.grand_prix}</p>
          <p><strong>First Pole:</strong> {data.first_pole.year} - {data.first_pole.grand_prix}</p>
          <p><strong>Last Pole:</strong> {data.last_pole.year} - {data.last_pole.grand_prix}</p>
          <p><strong>Teams:</strong> {data.teams.map(team => `${team.constructor} (${team.years})`).join(', ')}</p>
        </div>
      );
      setModalOpen(true);
    } catch (error) {
      console.error('Error fetching driver stats:', error);
      alert('Failed to fetch driver stats.');
    }
  };

  const handleTeamClick = async (constructor) => {
    try {
      const response = await axios.get(`http://localhost:8003/api/team_stats/${constructor}`);
      const data = response.data;
      setModalTitle(`Team Stats - ${constructor}`);
      setModalContent(
        <div>
          <p><strong>Total Entries:</strong>  {data.total_entries}</p>
          <p><strong>First Race:</strong> {data.first_race.year} - {data.first_race.grand_prix}</p>
          <p><strong>Last Race:</strong> {data.last_race.year} - {data.last_race.grand_prix}</p>
          <p><strong>Best Grid Position:</strong> {data.best_grid_position} ({data.best_grid_count} times)</p>
          <p><strong>Best Race Result:</strong> {data.best_race_result} ({data.best_race_count} times)</p>
          <p><strong>Best Championship Position:</strong> {data.best_championship_position} (Years: {data.best_championship_years})</p>
          <p><strong>Race Wins:</strong> {data.race_wins}</p>
          <p><strong>Pole Positions:</strong> {data.pole_positions}</p>
          <p><strong>Podiums:</strong> {data.podiums}</p>
          <p><strong>First Podium:</strong> {data.first_podium.year} - {data.first_podium.grand_prix}</p>
          <p><strong>Last Podium:</strong> {data.last_podium.year} - {data.last_podium.grand_prix}</p>
          <p><strong>First Win:</strong> {data.first_win.year} - {data.first_win.grand_prix}</p>
          <p><strong>Last Win:</strong> {data.last_win.year} - {data.last_win.grand_prix}</p>
          <p><strong>First Pole:</strong> {data.first_pole.year} - {data.first_pole.grand_prix}</p>
          <p><strong>Last Pole:</strong> {data.last_pole.year} - {data.last_pole.grand_prix}</p>
        </div>
      );
      setModalOpen(true);
    } catch (error) {
      console.error('Error fetching team stats:', error);
      alert('Failed to fetch team stats.');
    }
  };

  const tabs = [
    {
      name: 'Calendar',
      content: (
        <div>
          <h3 className="text-2xl font-semibold mb-4">Race Calendar {year}</h3>
          {error && <p className="text-red-500 mb-4">{error}</p>}
          {loading ? (
            <p>Loading...</p>
          ) : calendar.length === 0 ? (
            <p>No data available.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white rounded-lg shadow-md">
                <thead className="bg-gray-200">
                  <tr>
                    {calendar.length > 0 && Object.keys(calendar[0]).filter(key => !['PoleSitterId', 'WinnerId'].includes(key)).map(header => (
                      <th key={header} className="px-4 py-2 text-left">{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {calendar.map((race, index) => (
                    <tr key={index} className="border-t">
                      {Object.entries(race).filter(([key]) => !['PoleSitterId', 'WinnerId'].includes(key)).map(([key, value]) => {
                        if (key === 'GrandPrix' && race.Status === '✅ Completed') {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleGrandPrixClick(year, value)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else if (key === 'PoleSitter' && race.Status === '✅ Completed' && race.PoleSitterId) {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleDriverClick(race.PoleSitterId)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else if (key === 'Winner' && race.Status === '✅ Completed' && race.WinnerId) {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleDriverClick(race.WinnerId)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else {
                          return <td key={key} className="px-4 py-2">{String(value)}</td>;
                        }
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ),
    },
    {
      name: 'Championships',
      content: (
        <div>
          <h3 className="text-2xl font-semibold mb-4">Drivers Championship Standings {year}</h3>
          {error && <p className="text-red-500 mb-4">{error}</p>}
          {loading ? (
            <p>Loading...</p>
          ) : driversStandings.length === 0 ? (
            <p>No data available.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white rounded-lg shadow-md">
                <thead className="bg-gray-200">
                  <tr>
                    {driversStandings.length > 0 && Object.keys(driversStandings[0]).filter(key => !['driver_id'].includes(key)).map(header => (
                      <th key={header} className="px-4 py-2 text-left">{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {driversStandings.map((driver, index) => (
                    <tr key={index} className="border-t">
                      {Object.entries(driver).filter(([key]) => !['driver_id'].includes(key)).map(([key, value]) => {
                        if (key === 'Driver' && driver.driver_id) {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleDriverClick(driver.driver_id)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else if (key === 'Team') {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleTeamClick(value)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else {
                          return <td key={key} className="px-4 py-2">{String(value)}</td>;
                        }
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <h3 className="text-2xl font-semibold mt-6 mb-4">Constructors Championship Standings {year}</h3>
          {error && <p className="text-red-500 mb-4">{error}</p>}
          {loading ? (
            <p>Loading...</p>
          ) : constructorsStandings.length === 0 ? (
            <p>No data available.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white rounded-lg shadow-md">
                <thead className="bg-gray-200">
                  <tr>
                    {constructorsStandings.length > 0 && Object.keys(constructorsStandings[0]).filter(key => !['constructor_id'].includes(key)).map(header => (
                      <th key={header} className="px-4 py-2 text-left">{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {constructorsStandings.map((constructor, index) => (
                    <tr key={index} className="border-t">
                      {Object.entries(constructor).filter(([key]) => !['constructor_id'].includes(key)).map(([key, value]) => {
                        if (key === 'Team') {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleTeamClick(value)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else {
                          return <td key={key} className="px-4 py-2">{String(value)}</td>;
                        }
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ),
    },
    {
      name: 'Season Statistics',
      content: (
        <div>
          <h3 className="text-2xl font-semibold mb-4">Season Statistics {year}</h3>
          <div className="flex flex-wrap justify-center">
            <div className="w-1/2 p-4">
              <img src={`/images/driver_points_history_${year}.jpg`} alt="Driver Points History" className="w-full rounded-lg shadow-md" />
            </div>
            <div className="w-1/2 p-4">
              <img src={`/images/team_points_history_${year}.jpg`} alt="Team Points History" className="w-full rounded-lg shadow-md" />
            </div>
            <div className="w-1/2 p-4">
              <img src={`/images/driver_stats_${year}.jpg`} alt="Driver Stats" className="w-full rounded-lg shadow-md" />
            </div>
            <div className="w-1/2 p-4">
              <img src={`/images/constructor_stats_${year}.jpg`} alt="Team Stats" className="w-full rounded-lg shadow-md" />
            </div>
          </div>
        </div>
      ),
    },
    {
      name: 'Predictions',
      content: (
        <div>
          <div className="flex justify-between items-center mb-4">
            <div>
              <h3 className="text-2xl font-semibold">Next Grand Prix</h3>
              {nextRace ? (
                <div>
                  <p><strong>Grand Prix:</strong> {nextRace.GrandPrix}</p>
                  <p><strong>Date:</strong> {nextRace.Date}</p>
                  <p><strong>Circuit:</strong> {nextRace.Circuit}</p>
                </div>
              ) : (
                <p>No next race information available.</p>
              )}
            </div>
            <button
              className={`f1-tab-button px-4 py-2 rounded-lg font-semibold ${loading ? 'bg-gray-300 text-gray-500' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
              onClick={loadPredictions}
              disabled={loading}
            >
              {loading ? 'Loading...' : 'Load Predictions'}
            </button>
          </div>
          {predictionsLoaded ? (
            <div>
              <h3 className="text-2xl font-semibold mb-4">Qualifying Predictions</h3>
              {error && <p className="text-red-500 mb-4">{error}</p>}
              {loading ? (
                <p>Loading...</p>
              ) : qualiPredictions.length === 0 ? (
                <p>No data available.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white rounded-lg shadow-md">
                    <thead className="bg-gray-200">
                      <tr>
                        {qualiPredictions.length > 0 && Object.keys(qualiPredictions[0]).map(header => (
                          <th key={header} className="px-4 py-2 text-left">{header}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {qualiPredictions.map((prediction, index) => (
                        <tr key={index} className="border-t">
                          {Object.values(prediction).map((value, index) => (
                            <td key={index} className="px-4 py-2">{String(value)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <h3 className="text-2xl font-semibold mt-6 mb-4">Race Predictions</h3>
              {error && <p className="text-red-500 mb-4">{error}</p>}
              {loading ? (
                <p>Loading...</p>
              ) : racePredictions.length === 0 ? (
                <p>No data available.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white rounded-lg shadow-md">
                    <thead className="bg-gray-200">
                      <tr>
                        {racePredictions.length > 0 && Object.keys(racePredictions[0]).map(header => (
                          <th key={header} className="px-4 py-2 text-left">{header}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {racePredictions.map((prediction, index) => (
                        <tr key={index} className="border-t">
                          {Object.values(prediction).map((value, index) => (
                            <td key={index} className="px-4 py-2">{String(value)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            <p>Click the button to load predictions.</p>
          )}
        </div>
      ),
    },
    {
      name: 'Driver Results',
      content: (
        <div>
          <h3 className="text-2xl font-semibold mb-4">Race Results {year}</h3>
          {error && <p className="text-red-500 mb-4">{error}</p>}
          {loading ? (
            <p>Loading...</p>
          ) : raceResults.length === 0 ? (
            <p>No data available.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white rounded-lg shadow-md">
                <thead className="bg-gray-200">
                  <tr>
                    {raceResults.length > 0 && Object.keys(raceResults[0]).filter(key => key !== 'driver_id').map(header => (
                      <th key={header} className="px-4 py-2 text-left">{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {raceResults.map((result, index) => (
                    <tr key={index} className="border-t">
                      {Object.entries(result).filter(([key]) => key !== 'driver_id').map(([key, value]) => {
                        if (key === 'Driver' && result.driver_id) {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleDriverClick(result.driver_id)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else if (key === 'Team') {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleTeamClick(value)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else {
                          return <td key={key} className="px-4 py-2">{value}</td>;
                        }
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <h3 className="text-2xl font-semibold mt-6 mb-4">Qualifying Results {year}</h3>
          {error && <p className="text-red-500 mb-4">{error}</p>}
          {loading ? (
            <p>Loading...</p>
          ) : qualiResults.length === 0 ? (
            <p>No data available.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white rounded-lg shadow-md">
                <thead className="bg-gray-200">
                  <tr>
                    {qualiResults.length > 0 && Object.keys(qualiResults[0]).filter(key => key !== 'driver_id').map(header => (
                      <th key={header} className="px-4 py-2 text-left">{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {qualiResults.map((result, index) => (
                    <tr key={index} className="border-t">
                      {Object.entries(result).filter(([key]) => key !== 'driver_id').map(([key, value]) => {
                        if (key === 'Driver' && result.driver_id) {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleDriverClick(result.driver_id)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else if (key === 'Team') {
                          return (
                            <td key={key} className="px-4 py-2">
                              <span
                                className="text-white hover:underline cursor-pointer"
                                onClick={() => handleTeamClick(value)}
                              >
                                {value}
                              </span>
                            </td>
                          );
                        } else {
                          return <td key={key} className="px-4 py-2">{value}</td>;
                        }
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ),
    }
  ];

  return (
    <div className="max-w-6xl mx-auto mt-8 p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-3xl font-bold mb-6 text-white-800">F1 {year} Season</h2>
      <div className="flex space-x-4 mb-6">
        {tabs.map(tab => (
          <button
            key={tab.name}
            className={`f1-tab-button px-4 py-2 rounded-lg font-semibold ${activeTab === tab.name ? 'bg-red-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
            onClick={() => setActiveTab(tab.name)}
          >
            {tab.name}
          </button>
        ))}
      </div>
      {tabs.find(tab => tab.name === activeTab)?.content}
      <Modal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title={modalTitle}
      >
        {modalContent}
      </Modal>
    </div>
  );
}

export default F1;