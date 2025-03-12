
from typing import Optional, Dict, Any, List

from .tools import Tool
from .tool_metadata import ToolParameter, ToolDocumentation

try:
    import yfinance as yf
except ImportError:
    raise ImportError("`yfinance` not installed. Please install using `pip install yfinance`.")


class StockPriceTool(Tool):
    """Tool for getting current stock prices."""
    
    @property
    def name(self) -> str:
        return "StockPrice"
    
    @property
    def description(self) -> str:
        return "Get the current stock price for a given symbol"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type="string",
                description="The stock symbol (e.g., AAPL, MSFT, GOOG)",
                required=True
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: StockPrice "AAPL"',
            parameters=self.parameters
        )
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Get current stock price for the given symbol.
        
        Args:
            input_str: The stock symbol
            args: Optional structured arguments
            
        Returns:
            Current stock price information
        """
        symbol = input_str.strip()
        
        # Use args if provided, otherwise parse from input_str
        if args and "symbol" in args:
            symbol = args["symbol"]
        
        if not symbol:
            return "Error: Stock symbol is required"
        
        try:
            stock = yf.Ticker(symbol)
            # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
            current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            
            if current_price:
                currency = stock.info.get("currency", "USD")
                return f"Current price of {symbol}: ${current_price:.2f} {currency}"
            else:
                return f"Could not fetch current price for {symbol}"
        except Exception as e:
            return f"Error fetching stock price for {symbol}: {e}"


class CompanyInfoTool(Tool):
    """Tool for getting company information."""
    
    @property
    def name(self) -> str:
        return "CompanyInfo"
    
    @property
    def description(self) -> str:
        return "Get detailed company information for a given stock symbol"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type="string",
                description="The stock symbol (e.g., AAPL, MSFT, GOOG)",
                required=True
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: CompanyInfo "AAPL"',
            parameters=self.parameters
        )
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Get company information for the given symbol.
        
        Args:
            input_str: The stock symbol
            args: Optional structured arguments
            
        Returns:
            Company profile and overview
        """
        symbol = input_str.strip()
        
        # Use args if provided
        if args and "symbol" in args:
            symbol = args["symbol"]
        
        if not symbol:
            return "Error: Stock symbol is required"
        
        try:
            company_info_full = yf.Ticker(symbol).info
            if company_info_full is None:
                return f"Could not fetch company info for {symbol}"
            
            company_info_cleaned = {
                "Name": company_info_full.get("shortName"),
                "Symbol": company_info_full.get("symbol"),
                "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
                "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
                "Sector": company_info_full.get("sector"),
                "Industry": company_info_full.get("industry"),
                "Address": company_info_full.get("address1"),
                "City": company_info_full.get("city"),
                "State": company_info_full.get("state"),
                "Zip": company_info_full.get("zip"),
                "Country": company_info_full.get("country"),
                "EPS": company_info_full.get("trailingEps"),
                "P/E Ratio": company_info_full.get("trailingPE"),
                "52 Week Low": company_info_full.get("fiftyTwoWeekLow"),
                "52 Week High": company_info_full.get("fiftyTwoWeekHigh"),
                "50 Day Average": company_info_full.get("fiftyDayAverage"),
                "200 Day Average": company_info_full.get("twoHundredDayAverage"),
                "Website": company_info_full.get("website"),
                "Summary": company_info_full.get("longBusinessSummary"),
                "Analyst Recommendation": company_info_full.get("recommendationKey"),
                "Number Of Analyst Opinions": company_info_full.get("numberOfAnalystOpinions"),
                "Employees": company_info_full.get("fullTimeEmployees"),
            }
            
            formatted_output = "\n".join([f"{k}: {v}" for k, v in company_info_cleaned.items() if v is not None])
            return formatted_output
        except Exception as e:
            return f"Error fetching company information for {symbol}: {e}"


class StockHistoricalPricesTool(Tool):
    """Tool for getting historical stock prices."""
    
    @property
    def name(self) -> str:
        return "StockHistoricalPrices"
    
    @property
    def description(self) -> str:
        return "Get historical stock prices for a given symbol and time period. Use this to analyze stock performance over time."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type="string",
                description="The stock symbol (e.g., AAPL, MSFT, GOOG)",
                required=True
            ),
            ToolParameter(
                name="period",
                type="string",
                description="Time period to analyze (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max). Use '1mo' for one month, '3mo' for three months, etc.",
                required=False
            ),
            ToolParameter(
                name="interval",
                type="string",
                description="Data interval between points (1d, 5d, 1wk, 1mo, 3mo). Use '1d' for daily data.",
                required=False
            )
        ]
    
    @property
    def parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., AAPL)"
                },
                "period": {
                    "type": "string",
                    "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                    "description": "Time period to analyze"
                },
                "interval": {
                    "type": "string",
                    "enum": ["1d", "5d", "1wk", "1mo", "3mo"],
                    "description": "Data interval between points"
                }
            },
            "required": ["symbol"]
        }
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: StockHistoricalPrices {"symbol": "AAPL", "period": "1mo", "interval": "1d"}',
            parameters=self.parameters
        )
    
    @property
    def usage_examples(self) -> List[str]:
        return [
            # Example for last month's daily data
            'TOOL REQUEST: StockHistoricalPrices {"symbol": "AAPL", "period": "1mo", "interval": "1d"}',
            # Example for last 3 months weekly data
            'TOOL REQUEST: StockHistoricalPrices {"symbol": "MSFT", "period": "3mo", "interval": "1wk"}',
            # Example with just symbol (uses defaults)
            'TOOL REQUEST: StockHistoricalPrices {"symbol": "GOOG"}'
        ]
        
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Run the tool to fetch historical stock prices.
        
        Args:
            input_str: A string input (unused with args)
            args: Dictionary with 'symbol', 'period', and 'interval'
        
        Returns:
            A string containing the historical stock prices or an error message
        """
        import yfinance as yf
        import pandas as pd
        from datetime import datetime
        
        try:
            # Extract parameters
            if args:
                symbol = args.get("symbol", "").upper()
                period = args.get("period", "1mo")
                interval = args.get("interval", "1d")
            else:
                # Try to parse from input_str if no args
                return "Error: This tool requires JSON parameters. Please use format: {\"symbol\": \"AAPL\", \"period\": \"1mo\", \"interval\": \"1d\"}"
            
            # Validate the symbol first
            if not symbol:
                return "Error: No stock symbol provided. Please provide a valid stock symbol."
            
            # Verify the symbol exists
            try:
                ticker = yf.Ticker(symbol)
                # Try to get some basic info to verify the ticker exists
                # Different yfinance versions have different attributes
                try:
                    info = ticker.info
                    if not info or 'symbol' not in info:
                        common_symbols = "AAPL (Apple), MSFT (Microsoft), GOOG (Google), AMZN (Amazon), META (Facebook), TSLA (Tesla)"
                        return f"Error: Invalid or unknown stock symbol '{symbol}'. Please try one of these common symbols: {common_symbols}"
                except:
                    # Try to get history as a fallback check
                    test_history = ticker.history(period="1d")
                    if test_history.empty:
                        common_symbols = "AAPL (Apple), MSFT (Microsoft), GOOG (Google), AMZN (Amazon), META (Facebook), TSLA (Tesla)"
                        return f"Error: Invalid or unknown stock symbol '{symbol}'. Please try one of these common symbols: {common_symbols}"
            except Exception as e:
                common_symbols = "AAPL (Apple), MSFT (Microsoft), GOOG (Google), AMZN (Amazon), META (Facebook), TSLA (Tesla)"
                return f"Error: Could not validate stock symbol '{symbol}'. Error: {str(e)}. Please try one of these common symbols: {common_symbols}"
            
            # Validate period and interval
            valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
            valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
            
            if period not in valid_periods:
                return f"Error: Invalid period '{period}'. Valid periods are: {', '.join(valid_periods)}"
            
            if interval not in valid_intervals:
                return f"Error: Invalid interval '{interval}'. Valid intervals are: {', '.join(valid_intervals)}"
            
            # Get historical data
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return f"No historical data available for {symbol} with period={period} and interval={interval}. Try different parameters."
            
            # Format the data for readability
            formatted_data = []
            
            # Limit to last 5 data points to avoid overwhelming response
            hist_tail = hist.tail(5)
            
            for date, row in hist_tail.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                formatted_row = {
                    "Date": date_str,
                    "Open": round(row["Open"], 2),
                    "High": round(row["High"], 2),
                    "Low": round(row["Low"], 2),
                    "Close": round(row["Close"], 2),
                    "Volume": int(row["Volume"]) if "Volume" in row else "N/A"
                }
                formatted_data.append(formatted_row)
            
            # Calculate summary statistics for the full period
            if len(hist) >= 2:
                start_price = hist.iloc[0]["Close"]
                end_price = hist.iloc[-1]["Close"]
                price_change = end_price - start_price
                percent_change = (price_change / start_price) * 100
                
                high = hist["High"].max()
                low = hist["Low"].min()
                avg_volume = hist["Volume"].mean() if "Volume" in hist else "N/A"
                
                summary = (
                    f"Summary for {symbol} over {period} ({interval} intervals):\n"
                    f"- Starting price: ${start_price:.2f}\n"
                    f"- Current price: ${end_price:.2f}\n"
                    f"- Change: ${price_change:.2f} ({percent_change:.2f}%)\n"
                    f"- Highest price: ${high:.2f}\n"
                    f"- Lowest price: ${low:.2f}\n"
                )
                
                if avg_volume != "N/A":
                    summary += f"- Average trading volume: {int(avg_volume):,}\n"
                
                performance = ""
                if percent_change > 0:
                    performance = f"The stock has gained {percent_change:.2f}% over this period."
                elif percent_change < 0:
                    performance = f"The stock has lost {abs(percent_change):.2f}% over this period."
                else:
                    performance = "The stock price has remained stable over this period."
            else:
                summary = f"Insufficient data for {symbol} with the specified parameters to calculate summary statistics."
                performance = ""
            
            # Create a DataFrame for prettier output
            df_str = ""
            try:
                df = pd.DataFrame(formatted_data)
                df_str = df.to_string(index=False)
            except:
                # Fallback to simple string representation
                df_str = str(formatted_data)
            
            # Combine results
            result = (
                f"Historical price data for {symbol} (showing last 5 data points):\n\n"
                f"{df_str}\n\n"
                f"{summary}\n"
                f"{performance}"
            )
            
            return result
        
        except Exception as e:
            error_msg = str(e)
            return f"Error retrieving historical stock prices for symbol '{symbol}': {error_msg}. Please verify the symbol and parameters."


class StockFundamentalsTool(Tool):
    """Tool for getting stock fundamentals."""
    
    @property
    def name(self) -> str:
        return "StockFundamentals"
    
    @property
    def description(self) -> str:
        return "Get fundamental financial data for a given stock symbol"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type="string",
                description="The stock symbol (e.g., AAPL, MSFT, GOOG)",
                required=True
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: StockFundamentals "AAPL"',
            parameters=self.parameters
        )
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Get fundamental data for the given stock symbol.
        
        Args:
            input_str: The stock symbol
            args: Optional structured arguments
            
        Returns:
            Fundamental financial data
        """
        symbol = input_str.strip()
        
        # Use args if provided
        if args and "symbol" in args:
            symbol = args["symbol"]
        
        if not symbol:
            return "Error: Stock symbol is required"
        
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("forwardPE", "N/A"),
                "pb_ratio": info.get("priceToBook", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "beta": info.get("beta", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                "profit_margins": info.get("profitMargins", "N/A"),
                "revenue_growth": info.get("revenueGrowth", "N/A"),
                "gross_margins": info.get("grossMargins", "N/A"),
                "ebitda_margins": info.get("ebitdaMargins", "N/A"),
            }
            
            formatted_output = "\n".join([f"{k}: {v}" for k, v in fundamentals.items() if v != "N/A"])
            return formatted_output
        except Exception as e:
            return f"Error getting fundamentals for {symbol}: {e}"


class FinancialStatementsTool(Tool):
    """Tool for getting financial statements."""
    
    @property
    def name(self) -> str:
        return "FinancialStatements"
    
    @property
    def description(self) -> str:
        return "Get income statements and balance sheets for a given stock symbol"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="symbol",
                type="string",
                description="The stock symbol (e.g., AAPL, MSFT, GOOG)",
                required=True
            ),
            ToolParameter(
                name="statement_type",
                type="string",
                description="Type of statement (income, balance_sheet, cash_flow)",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: FinancialStatements {"symbol": "AAPL", "statement_type": "income"}',
            parameters=self.parameters
        )
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Get financial statements for the given stock symbol.
        
        Args:
            input_str: The stock symbol
            args: Optional structured arguments including statement_type
            
        Returns:
            Financial statement data
        """
        symbol = input_str.strip()
        statement_type = "income"  # Default to income statement
        
        # Use args if provided
        if args:
            if "symbol" in args:
                symbol = args["symbol"]
            if "statement_type" in args:
                statement_type = args["statement_type"]
        
        if not symbol:
            return "Error: Stock symbol is required"
        
        try:
            stock = yf.Ticker(symbol)
            
            if statement_type.lower() == "income":
                financials = stock.financials
                statement_name = "Income Statement"
            elif statement_type.lower() == "balance_sheet":
                financials = stock.balance_sheet
                statement_name = "Balance Sheet"
            elif statement_type.lower() == "cash_flow":
                financials = stock.cashflow
                statement_name = "Cash Flow Statement"
            else:
                return f"Invalid statement type: {statement_type}. Choose from: income, balance_sheet, cash_flow"
            
            if financials.empty:
                return f"No {statement_name.lower()} data available for {symbol}"
            
            # Convert the dataframe to a more readable format
            # Get the most recent 2 columns (years) to avoid overwhelming responses
            recent_years = financials.columns.tolist()[:2]
            result_data = {}
            
            for item in financials.index:
                item_name = str(item)
                result_data[item_name] = {}
                
                for year in recent_years:
                    # Format the value as a readable number
                    value = financials.loc[item, year]
                    if isinstance(value, (int, float)):
                        formatted_value = f"${value:,.0f}"
                    else:
                        formatted_value = str(value)
                    
                    # Use the year as a string key
                    year_str = str(year.year) if hasattr(year, 'year') else str(year)
                    result_data[item_name][year_str] = formatted_value
            
            output = [f"{statement_name} for {symbol}:"]
            for item, years_data in result_data.items():
                years_str = ", ".join([f"{year}: {value}" for year, value in years_data.items()])
                output.append(f"{item}: {years_str}")
            
            return "\n".join(output)
        except Exception as e:
            return f"Error fetching financial statements for {symbol}: {e}"


class YFinanceToolkit:
    """
    A collection of YFinance tools for stock analysis.
    
    Provides tools for retrieving stock prices, company information,
    historical data, and financial statements.
    """
    
    def __init__(self, 
                 enable_stock_price: bool = True,
                 enable_company_info: bool = True,
                 enable_historical_prices: bool = True,
                 enable_fundamentals: bool = True,
                 enable_financials: bool = True):
        """
        Initialize the YFinance toolkit.
        
        Args:
            enable_stock_price: Whether to enable the stock price tool
            enable_company_info: Whether to enable the company info tool
            enable_historical_prices: Whether to enable the historical prices tool
            enable_fundamentals: Whether to enable the fundamentals tool
            enable_financials: Whether to enable the financial statements tool
        """
        self.tools = []
        
        if enable_stock_price:
            self.tools.append(StockPriceTool())
        
        if enable_company_info:
            self.tools.append(CompanyInfoTool())
        
        if enable_historical_prices:
            self.tools.append(StockHistoricalPricesTool())
        
        if enable_fundamentals:
            self.tools.append(StockFundamentalsTool())
        
        if enable_financials:
            self.tools.append(FinancialStatementsTool())
    
    def get_tools(self) -> List[Tool]:
        """
        Get all the enabled YFinance tools.
        
        Returns:
            List of enabled Tool instances
        """
        return self.tools 