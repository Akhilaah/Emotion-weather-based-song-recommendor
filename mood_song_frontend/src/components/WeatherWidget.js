import React, { useEffect, useState } from "react";
import axios from "axios";

function WeatherWidget() {
  const [weather, setWeather] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        // Get user's current location
        navigator.geolocation.getCurrentPosition(async (position) => {
          const { latitude, longitude } = position.coords;

          const API_KEY = "ceac93a247353ad23cf698fe6c29531d"; // Replace this
          const url = `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${API_KEY}&units=metric`;

          const response = await axios.get(url);
          setWeather(response.data);
        });
      } catch (err) {
        setError("Failed to fetch weather. Please allow location access.");
      }
    };

    fetchWeather();
  }, []);

  if (error) return <p>{error}</p>;
  if (!weather) return <p>Loading weather...</p>;

  return (
    <div
      style={{
        background: "rgba(255,255,255,0.1)",
        padding: "15px",
        borderRadius: "10px",
        marginBottom: "20px",
        color: "#fff",
        textAlign: "center",
      }}
    >
      <h3>🌤️ Current Weather</h3>
      <p style={{ fontSize: "18px" }}>
        {weather.name} — {weather.weather[0].main}
      </p>
      <p style={{ fontSize: "16px" }}>
        🌡️ {weather.main.temp}°C | 💧 {weather.main.humidity}% humidity
      </p>
    </div>
  );
}

export default WeatherWidget;
