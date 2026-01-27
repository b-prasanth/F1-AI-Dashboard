import React, { useState, useEffect } from 'react';
import './ChampionshipsTab.css'; // Import the CSS file

const ChampionshipsTab = () => {
  const [driverStandings, setDriverStandings] = useState([]);
  const [constructorStandings, setConstructorStandings] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStandings = async () => {
      try {
        const driverResponse = await fetch('/api/standings/drivers');
        if (!driverResponse.ok) {
          throw new Error(`HTTP error! status: ${driverResponse.status}`);
        }
        const driverData = await driverResponse.json();
        setDriverStandings(driverData);

        const constructorResponse = await fetch('/api/standings/constructors');
        if (!constructorResponse.ok) {
          throw new Error(`HTTP error! status: ${constructorResponse.status}`);
        }
        const constructorData = await constructorResponse.json();
        setConstructorStandings(constructorData);
      } catch (e) {
        setError(e.message);
      }
    };

    fetchStandings();
  }, []);

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="championships-container">
      <div className="driver-standings">
        <h2>Driver Standings</h2>
        <table>
          <thead>
            <tr>
              {driverStandings.length > 0 && Object.keys(driverStandings[0]).map(header => (
                <th key={header}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {driverStandings.map((driver, index) => (
              <tr key={index}>
                {Object.values(driver).map((value, index) => (
                  <td key={index}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="constructor-standings">
        <h2>Constructor Standings</h2>
        <table>
          <thead>
            <tr>
              {constructorStandings.length > 0 && Object.keys(constructorStandings[0]).map(header => (
                <th key={header}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {constructorStandings.map((constructor, index) => (
              <tr key={index}>
                {Object.values(constructor).map((value, index) => (
                  <td key={index}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ChampionshipsTab;
