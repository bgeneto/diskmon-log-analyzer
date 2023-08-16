# DiskMon Log Analyzer

This is a Streamlit (Python) application that analyzes the log file generated by DiskMon, a tool that logs and displays all hard disk activity on a Windows system. The application provides a variety of insights into disk performance, including:

* **Total read and write requests**
* **Percentage of random and sequential requests**
* **Most frequently requested block size**
* **Average access time**
* **Distribution of request lengths**

The application is easy to use. Simply upload your DiskMon log file to the application and it will automatically analyze the data and generate the results. You can then explore the results in detail by clicking on the various charts and tables.

## How to use the application

1. Navigate to the [DiskMon Log Analyzer HomePage](https://diskmon-log-visualizer.streamlit.app)
2. Upload your DiskMon log file to the application.
3. The application will automatically analyze the data and generate the results.
4. Explore the results by clicking on the various charts and tables.
5. You can also rename the disks in your system by clicking on the "Name your disks" button.

## Results 

The following is just an example of the results that the application can generate. 
(graphical results not shown)

|                       | Value      |
| --------------------- | :--------- |
| Total monitoring time | 80 minutes |
| Total read requests   | 426008     |
| Total write requests  | 195896     |
| Total requests        | 621904     |
| Total percent read    | 68.50%     |
| Total percent write   | 31.50%     |
| Total RND requests    | 95.67%     |
| Total SEQ requests    | 4.33%      |
| Total data read       | 11.15 GB   |
| Total data write      | 6.21 GB    |
| Total data size       | 17.36 GB   |
| Min. read request     | 0.5 KB     |
| Avg. read request     | 27.5 KB    |
| Max. read request     | 4096 KB    |
| Min. write request    | 0.5 KB     |
| Avg. write request    | 33.2 KB    |
| Max. write request    | 2176 KB    |


## Contact

If you have any questions about the application, please contact the author.

## License

The application is released under the MIT license.

## Disclaimer

The application is provided as-is and the author is not responsible for any damages that may result from its use.

