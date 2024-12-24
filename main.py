import numpy as np


def initialize_components(series, season_length):
    """Инициализация базовых компонентов: уровня, тренда и сезонности."""
    series = np.array(series)
    level = series[:season_length].mean()
    trend = (series[season_length:2 * season_length].mean() - series[:season_length].mean()) / season_length
    seasonality = [series[i] / level for i in range(season_length)]
    return level, trend, np.array(seasonality)


def holt_winters_forecasting(series, alpha1, alpha2, alpha3, season_length, forecast_periods):
    """Реализация модели Хольта-Уинтерса."""
    series = np.array(series)
    n = len(series)
    level, trend, seasonality = initialize_components(series, season_length)

    levels = [level]
    trends = [trend]
    seasonalities = list(seasonality)

    for t in range(n):
        # Индекс сезонности с учетом цикла
        season_idx = t % season_length
        # Обновляем уровень
        level = alpha1 * (series[t] / seasonalities[season_idx]) + (1 - alpha1) * (levels[-1] + trends[-1])
        # Обновляем тренд
        trend = alpha2 * (level - levels[-1]) + (1 - alpha2) * trends[-1]
        # Обновляем сезонность
        seasonalities[season_idx] = alpha3 * (series[t] / level) + (1 - alpha3) * seasonalities[season_idx]

        # Сохраняем обновления
        levels.append(level)
        trends.append(trend)

    # Генерация прогнозов
    forecasts = []
    for k in range(1, forecast_periods + 1):
        season_idx = (n + k - 1) % season_length
        forecast = (levels[-1] + k * trends[-1]) * seasonalities[season_idx]
        forecasts.append(forecast)

    return levels, trends, seasonalities, forecasts


# Пример использования
if __name__ == "__main__":
    # Пример временного ряда
    data = [120, 135, 150, 170, 130, 145, 160, 180, 140, 155, 170, 190]  # Годовые данные (4 сезона)
    season_length = 4  # Длина сезона (4 квартала)
    forecast_periods = 4  # Прогноз на 1 год (4 квартала)

    # Задаем сглаживающие коэффициенты
    alpha1, alpha2, alpha3 = 0.5, 0.3, 0.2

    # Запуск модели
    levels, trends, seasonalities, forecasts = holt_winters_forecasting(
        data, alpha1, alpha2, alpha3, season_length, forecast_periods
    )

    # Вывод результатов
    print("Уровни (Levels):", ', '.join([str(np.round(x, 2)) for x in levels]))
    print("Тренды (Trends):", ', '.join([str(np.round(x, 2)) for x in trends]))
    print("Сезонности (Seasonalities):", ', '.join([str(np.round(x, 2)) for x in seasonalities]))
    print("Прогнозы (Forecasts):", ', '.join([str(np.round(x, 2)) for x in forecasts]))
