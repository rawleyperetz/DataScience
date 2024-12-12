#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)
library(lubridate)
library(tidyquant)
library(ggplot2)
library(dplyr)
library(forecast)
library(tidyr)
#library(xts)
library(zoo)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Stock Market Analysis"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
          textInput("sname", "Enter Stock Name", "", placeholder = "AAPL"),
          
            selectInput(
              inputId = "AnalMethod",
              label = "Analysis Method",
              choices = c("naive", "average", "drift method")
            ),
          
          sliderInput(inputId = "nyear", label = "Years", min = 1, max = 10, step = 1, value = 1),
        sliderInput(inputId = "forecastdays", label = "Days (Forecast)", min = 1, max = 100, step = 1, value = 30)
    ),

        # Show a plot of the generated distribution
        mainPanel(
          uiOutput("dynamicTitle"),
          uiOutput("dynamicDateRange"),
          plotOutput("stockPlot")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  output$dynamicTitle <- renderUI({
    h1(input$sname)
  })
  
  
  # Render dynamic date range
  output$dynamicDateRange <- renderUI({
    start_date <- as.Date(Sys.Date()) - years(input$nyear)
    end_date <- Sys.Date()
    h4(paste("From", start_date, "To", end_date))
  })

    output$stockPlot <- renderPlot({
        # generate bins based on input$bins from ui.R
      start_date <- as.Date(Sys.Date()) - years(input$nyear)
      end_date <- Sys.Date()
      # stock_data <- tq_get(input$sname, from = start_date, to = end_date)
      
      stock_data <- tryCatch(
        {
          tq_get(input$sname, from = start_date, to = end_date) %>% select("date", "adjusted")# %>% mutate(date = as.Date(date)) %>% select("date", "adjusted")
        },
        error = function(e) {
          NULL  # Return NULL if the data fetch fails
        }
      )
      
      # Check if data was successfully fetched
      if (is.null(stock_data) || nrow(stock_data) == 0) {
        return(plot(1, type = "n", xlab = "", ylab = "", main = "No Data Available"))
      }
      #stock_xts <- window(as.ts(stock_data), start = years(end_date), end = c(years(start_date),1)) #xts(stock_data$adjusted, order.by = stock_data$date)
      # Ensure complete date range
      complete_data <- stock_data %>%
        complete(date = seq(min(date), max(date), by = "day")) %>%
        mutate(adjusted = zoo::na.locf(adjusted, na.rm = FALSE))  # Fill NA with last observation
      
      # Prepare time series and naive forecast
      stock_ts <- ts(complete_data$adjusted, frequency = 252)
      
        if(input$AnalMethod == "naive"){
          naive_forecast <- naive(stock_ts, h = input$forecastdays)
          
          forecast_tibble <- tibble(
            date = seq(
              from = max(complete_data$date) + 1,
              by = "day",
              length.out = length(naive_forecast$mean)
            ),
            adjusted = as.numeric(naive_forecast$mean)
          )
          
          # Combine the original data and forecast
          plot_data <- bind_rows(
            complete_data %>% select(date, adjusted),
            forecast_tibble
          )
          
          # Plot with ggplot2
          ggplot(plot_data, aes(x = date, y = adjusted)) +
            geom_line(color = "blue") +  # Original data
            geom_line(data = forecast_tibble, aes(x = date, y = adjusted), color = "red", linetype = "dashed") +  # Forecast
            labs(
              title = "Stock Prices with Naive Forecast",
              x = "Date",
              y = "Adjusted Price",
              caption = "Blue: Original Data, Red Dashed: Naive Forecast"
            ) +
            theme_minimal()
          # autoplot(ts(stock_data)) + autolayer(naive(ts(stock_data), h=11), series = "Naive", PI=FALSE) +
          #   xlab("Date") + ylab("Adjusted") +
          #   guides(colour=guide_legend(title="Forecast"))
            # stock_data %>%
            # ggplot(aes(x = date, y = adjusted)) +
            # geom_line(color = "blue") +
            # xlab("Date") + ylab("Adjusted Price") +
            # theme_minimal()
        } else if (input$AnalMethod == "average"){
          avg_forecast <- meanf(stock_ts, h = input$forecastdays)
          
          forecast_tibble <- tibble(
            date = seq(
              from = max(complete_data$date) + 1,
              by = "day",
              length.out = length(avg_forecast$mean)
            ),
            adjusted = as.numeric(avg_forecast$mean)
          )
          
          # Combine the original data and forecast
          plot_data <- bind_rows(
            complete_data %>% select(date, adjusted),
            forecast_tibble
          )
          
          # Plot with ggplot2
          ggplot(plot_data, aes(x = date, y = adjusted)) +
            geom_line(color = "blue") +  # Original data
            geom_line(data = forecast_tibble, aes(x = date, y = adjusted), color = "red", linetype = "dashed") +  # Forecast
            labs(
              title = "Stock Prices with Average Forecast",
              x = "Date",
              y = "Adjusted Price",
              caption = "Blue: Original Data, Red Dashed: Average Forecast"
            ) +
            theme_minimal()
          # autoplot(stock_xts) + autolayer(meanf(as.ts(stock_xts), h=11), series = "Average", PI=FALSE) +
          #   xlab("Date") + ylab("Adjusted") +
          #   guides(colour=guide_legend(title="Forecast"))
        } else if (input$AnalMethod == "drift method"){
          drift_forecast <- rwf(stock_ts, drift = TRUE ,h = input$forecastdays)
          
          forecast_tibble <- tibble(
            date = seq(
              from = max(complete_data$date) + 1,
              by = "day",
              length.out = length(drift_forecast$mean)
            ),
            adjusted = as.numeric(drift_forecast$mean)
          )
          
          # Combine the original data and forecast
          plot_data <- bind_rows(
            complete_data %>% select(date, adjusted),
            forecast_tibble
          )
          
          # Plot with ggplot2
          ggplot(plot_data, aes(x = date, y = adjusted)) +
            geom_line(color = "blue") +  # Original data
            geom_line(data = forecast_tibble, aes(x = date, y = adjusted), color = "red", linetype = "dashed") +  # Forecast
            labs(
              title = "Stock Prices with Drift Forecast",
              x = "Date",
              y = "Adjusted Price",
              caption = "Blue: Original Data, Red Dashed: Drift Forecast"
            ) +
            theme_minimal()
          # autoplot(stock_xts) + autolayer(rwf(as.ts(stock_xts), h=11), series = "Average", PI=FALSE) +
          #   xlab("Date") + ylab("Adjusted") +
          #   guides(colour=guide_legend(title="Forecast"))
        }
      else {
          # Placeholder for "not naive" logic
          plot(1, type = "n", xlab = "", ylab = "", main = "Analysis Method Not Implemented")
        }
        
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
