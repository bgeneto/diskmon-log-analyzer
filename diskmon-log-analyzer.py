#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""DiskMon Log Analyzer.

This script analyzes the Diskmon.LOG file to provide insights and statistics about the disk monitoring data.

Author:   b g e n e t o @ g m a i l . c o m
History:  v1.0.0 Initial release
          v1.0.1 ?
Modified: 20230816
Usage:
    $ streamlit run diskmon-log-analyzer.py
"""

import base64

import pandas as pd
import plotly.express as px
import streamlit as st

diskmon = """
DiskMon is an application that logs and displays all hard disk activity on a Windows system.
For more information, please check [this page](https://learn.microsoft.com/en-us/sysinternals/downloads/diskmon).

You can download DiskMon executable from [here](https://download.sysinternals.com/files/DiskMon.zip).
"""

summary = """
This page enables you to determine the actual percentage of random reading or writing,
identify the most frequently requested block size, calculate the average access time to files,
and assess various other factors. These factors are valuable for detecting performance issues
related to the disk I/O. Additionally, they assist in selecting the optimal SSD for your machine based
on your own usage pattern instead of relying solely in generic benchmarks results.
"""

python_svg = """
<svg width="800px" height="800px" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M31.885 16c-8.124 0-7.617 3.523-7.617 3.523l.01 3.65h7.752v1.095H21.197S16 23.678 16 31.876c0 8.196 4.537 7.906 4.537 7.906h2.708v-3.804s-.146-4.537 4.465-4.537h7.688s4.32.07 4.32-4.175v-7.019S40.374 16 31.885 16zm-4.275 2.454c.771 0 1.395.624 1.395 1.395s-.624 1.395-1.395 1.395a1.393 1.393 0 0 1-1.395-1.395c0-.771.624-1.395 1.395-1.395z" fill="url(#a)"/><path d="M32.115 47.833c8.124 0 7.617-3.523 7.617-3.523l-.01-3.65H31.97v-1.095h10.832S48 40.155 48 31.958c0-8.197-4.537-7.906-4.537-7.906h-2.708v3.803s.146 4.537-4.465 4.537h-7.688s-4.32-.07-4.32 4.175v7.019s-.656 4.247 7.833 4.247zm4.275-2.454a1.393 1.393 0 0 1-1.395-1.395c0-.77.624-1.394 1.395-1.394s1.395.623 1.395 1.394c0 .772-.624 1.395-1.395 1.395z" fill="url(#b)"/><defs><linearGradient id="a" x1="19.075" y1="18.782" x2="34.898" y2="34.658" gradientUnits="userSpaceOnUse"><stop stop-color="#387EB8"/><stop offset="1" stop-color="#366994"/></linearGradient><linearGradient id="b" x1="28.809" y1="28.882" x2="45.803" y2="45.163" gradientUnits="userSpaceOnUse"><stop stop-color="#FFE052"/><stop offset="1" stop-color="#FFC331"/></linearGradient></defs></svg>
"""
pandas_svg = """
<svg data-name="Layer 1" version="1.1" viewBox="0 0 210.21 280.43" xmlns="http://www.w3.org/2000/svg" xmlns:cc="http://creativecommons.org/ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<metadata><rdf:RDF><cc:Work rdf:about=""><dc:format>image/svg+xml</dc:format><dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/></cc:Work></rdf:RDF></metadata>
<defs><style>.cls-1{fill:#130754;}.cls-2{fill:#48e5ac;}.cls-3{fill:#e70488;}</style></defs>
<rect class="cls-1" x="74.51" y="43.03" width="24.09" height="50.02"/>
<rect class="cls-1" x="74.51" y="145.78" width="24.09" height="50.02"/>
<rect class="cls-2" x="74.51" y="107.65" width="24.09" height="23.6" fill="#ffca00"/>
<rect class="cls-1" x="35.81" y="84.15" width="24.09" height="166.27"/>
<rect class="cls-1" x="112.41" y="187.05" width="24.09" height="50.02"/>
<rect class="cls-1" x="112.41" y="84.21" width="24.09" height="50.02"/>
<rect class="cls-3" x="112.41" y="148.84" width="24.09" height="23.6"/>
<rect class="cls-1" x="150.3" y="30" width="24.09" height="166.27"/>
</svg>
"""

usage = """
Just run `Diskmon64.exe` ([link](https://download.sysinternals.com/files/DiskMon.zip)) and let it monitor your disk activity for a while.
Once you are done tracing the I/O activity you want, save the log file, compress it (optional)
and upload it to this page. The page script &mdash; written in Python{python_svg}using Pandas{pandas_svg} &mdash; will analyze the log file and
provide you with the results.
> **Note:** tested with DiskMon for Windows v2.02 on Windows 11.
"""


def render_svg(svg, width="100%", height="100%") -> str:
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = (
        rf'<img width={width} height={height} src="data:image/svg+xml;base64,{b64}"/>'
    )
    return html


@st.cache_data
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read the uploaded file using pandas."""

    # check if the uploaded file is compressed
    compression = None
    if uploaded_file.name.endswith(".zip"):
        compression = "zip"
    elif uploaded_file.name.endswith(".gz"):
        compression = "gzip"
    elif uploaded_file.name.endswith(".bz2"):
        compression = "bz2"

    # Read the uploaded file using pandas
    data = pd.read_csv(
        uploaded_file,
        sep="\t",
        header=None,
        names=column_names,
        index_col="Index",
        compression=compression,
    )

    # Convert "Time" column to float
    data["Time"] = data["Time"].astype(float)

    # Convert "Disks" to string so plotly can recognize it as categorical (not numerical, continuous)
    data["Disks"] = "Disk " + data["Disks"].astype(str)

    # Check if duration is present and reliable, else calculate it
    if data["Duration (ms)"].sum() < 0.05 * data["Time"].sum():
        # Calculate the Duration (time delta) in milliseconds based on the "Time" column
        data["Duration (ms)"] = data["Time"].diff() * 1000

    # Classify the operation type based on the "Sector" and "Length" columns
    data["Type"] = (data["Sector"].diff() == data["Length"]).replace(
        {True: "SEQ", False: "RND"}
    )
    return data


@st.cache_data
def update_disks_names(data: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Change disk name."""
    for old_name, new_name in names.items():
        data.loc[data["Disks"] == old_name, "Disks"] = new_name

    return data


def dict_to_markdown_table(data_dict, header={"Metric": "Value"}):
    """Convert a dictionary to a markdown table."""
    first_key, first_value = next(iter(header.items()))
    markdown_table = f"| {first_key} | {first_value} |\n"
    markdown_table += "|--------|-------|\n"
    for key, value in data_dict.items():
        markdown_table += f"| {key} | {value} |\n"

    return markdown_table


@st.cache_data
def log_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Show totals."""

    log_summary = {}

    # Compute the total monitoring time by taking the max value from the "Time" column
    log_summary["Total monitoring time"] = "{:.2g} minutes".format(
        data["Time"].iloc[-1] / 60
    )

    # For example, let's calculate the average time for each operation (not accurate because duration is not always available)
    # totals["Average access time"] = "{:.2g} ms".format(data["Duration (ms)"].mean())

    # total read and write requests
    log_summary["Total read requests"] = data[data["Request"] == "Read"].shape[0]
    log_summary["Total write requests"] = data[data["Request"] == "Write"].shape[0]

    # total requests
    log_summary["Total requests"] = data.shape[0]

    # total percentage of read and write requests
    log_summary["Total percent read"] = "{:.2f}%".format(
        log_summary["Total read requests"] / log_summary["Total requests"] * 100
    )
    log_summary["Total percent write"] = "{:.2f}%".format(
        log_summary["Total write requests"] / log_summary["Total requests"] * 100
    )

    # total percentage SEQ and RND requests
    log_summary["Total RND requests"] = "{:.2f}%".format(
        data[data["Type"] == "RND"].shape[0] / log_summary["Total requests"] * 100
    )
    log_summary["Total SEQ requests"] = "{:.2f}%".format(
        data[data["Type"] == "SEQ"].shape[0] / log_summary["Total requests"] * 100
    )

    # total data read in Gbytes
    log_summary["Total data read"] = "{:.2f} GB".format(
        sector_size * data[data["Request"] == "Read"]["Length"].sum() / toGB
    )

    # total data written in Gbytes
    log_summary["Total data write"] = "{:.2f} GB".format(
        sector_size * data[data["Request"] == "Write"]["Length"].sum() / toGB
    )

    # total data size in Gbytes
    log_summary["Total data size"] = "{:.2f} GB".format(
        sector_size * data["Length"].sum() / toGB
    )

    # min and max readrequests in KB
    log_summary["Min. read request"] = "{:.1f} KB".format(
        sector_size * data[data["Request"] == "Read"]["Length"].min() / toKB
    )

    # avg read request in KB
    log_summary["Avg. read request"] = "{:.1f} KB".format(
        sector_size * data[data["Request"] == "Read"]["Length"].mean() / toKB
    )

    # max read requests in KB
    number = sector_size * data[data["Request"] == "Read"]["Length"].max() / toKB
    formatted_number = (
        "{:.1f}".format(number) if number % 1 else "{:.0f}".format(number)
    )
    log_summary["Max. read request"] = formatted_number + " KB"

    # min and max write requests in KB
    log_summary["Min. write request"] = "{:.1f} KB".format(
        sector_size * data[data["Request"] == "Write"]["Length"].min() / toKB
    )

    # avg write request in KB
    log_summary["Avg. write request"] = "{:.1f} KB".format(
        sector_size * data[data["Request"] == "Write"]["Length"].mean() / toKB
    )

    # max write requests in KB
    number = sector_size * data[data["Request"] == "Write"]["Length"].max() / toKB
    formatted_number = (
        "{:.1f}".format(number) if number % 1 else "{:.0f}".format(number)
    )
    log_summary["Max. write request"] = formatted_number + " KB"

    return pd.DataFrame.from_dict(log_summary, orient="index", columns=["Value"])


def plot_summary(data: pd.DataFrame):
    # Total requests in GB
    # --------------------------------------
    df = (
        (sector_size * data.groupby(["Disks", "Request"])["Length"].sum() / toGB)
        .to_frame()
        .reset_index()
    )

    # Calculate percentage within each disk group
    df["Percent"] = df["Length"] / df.groupby("Disks")["Length"].transform("sum") * 100

    df.rename(columns={"Length": "Length (GB)"}, inplace=True)

    # Create a plotly bar chart
    fig = px.bar(
        df,
        x="Disks",
        y="Length (GB)",
        title="Total Requested Data Size",
        color="Request",
        barmode="group",
        text="Percent",
    )

    # Annotate the bars with percentage values
    fig.update_traces(texttemplate="%{text:.3s}%", textposition="inside")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data"):
        st.dataframe(df)

    # RND and SEQ requested data size in GB
    # --------------------------------------
    df = (
        (sector_size * data.groupby(["Disks", "Type"])["Length"].sum() / toGB)
        .to_frame()
        .reset_index()
    )

    # Calculate percentage within each disk group
    df["Percent"] = df["Length"] / df.groupby("Disks")["Length"].transform("sum") * 100

    df.rename(columns={"Length": "Length (GB)"}, inplace=True)

    # Create a plotly bar chart
    fig = px.bar(
        df,
        x="Disks",
        y="Length (GB)",
        title="Data Size by Access Type (Random/Sequential)",
        color="Type",
        barmode="group",
        text="Percent",
    )

    # Annotate the bars with percentage values
    fig.update_traces(texttemplate="%{text:.3s}%", textposition="inside")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data"):
        st.write(
            '> **Note:** The percentage displayed on this chart is distinct from the "Total percent read" and "Total percent write" indicated in '
            + "the summary table above. This distinction arises because the table presents the percentage of requests, whereas the chart illustrates "
            + "the percentage of data. Because the length of each request can vary, numerous (small) requests may lead to a higher number of requests "
            + "but lower amount of data being read or written."
        )
        st.dataframe(df)

    # RND and SEQ requested data by request type per disk
    # --------------------------------------
    disks_names = sorted(data["Disks"].unique().tolist())
    for disk_name in disks_names:
        df = (
            (
                sector_size
                * data[data["Disks"] == disk_name]
                .groupby(["Request", "Type"])["Length"]
                .sum()
                / toGB
            )
            .to_frame()
            .reset_index()
        )

        # Calculate percentage within each disk group
        df["Percent"] = (
            df["Length"] / df.groupby("Type")["Length"].transform("sum") * 100
        )

        df.rename(columns={"Length": "Length (GB)"}, inplace=True)

        # Create a plotly bar chart
        fig = px.bar(
            df,
            x="Type",
            y="Length (GB)",
            title=f"Access Type (RND/SEQ) by Request Type (R/W) - {disk_name}",
            color="Request",
            barmode="group",
            text="Percent",
        )

        # Annotate the bars with percentage values
        fig.update_traces(texttemplate="%{text:.3s}%", textposition="inside")

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show data"):
            st.dataframe(df)


def outlier_thresholds_iqr(data: pd.DataFrame, col_name: str, lth=0.05, uth=0.95):
    """
    Calculates the lower and upper outlier thresholds using the interquartile range (IQR) method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        col_name (str): The column name from which to calculate the thresholds.
        th1 (float): The lower percentile for the quartile calculation. Defaults to 0.05.
        th3 (float): The upper percentile for the quartile calculation. Defaults to 0.95.

    Returns:
        float: The lower outlier threshold.
        float: The upper outlier threshold.
    """
    quartile1 = data[col_name].quantile(lth)
    quartile3 = data[col_name].quantile(uth)
    iqr = quartile3 - quartile1
    upper_limit = quartile3 + 1.5 * iqr
    lower_limit = quartile1 - 1.5 * iqr
    return lower_limit, upper_limit


def show_access_time(data: pd.DataFrame):
    """Compute disk access time."""
    # Remove outliers
    _, ub = outlier_thresholds_iqr(data, "Duration (ms)", 0.01, 0.99)

    # Filter the DataFrame based on quartiles
    fdf = data[(data["Duration (ms)"] > 0) & (data["Duration (ms)"] <= ub)]

    # average access time by disk
    df = fdf.groupby(["Disks"])[
        "Duration (ms)"
    ].mean()  # Reset index to access "Disks" as a column

    # Create Plotly bar plot
    fig = px.bar(
        df.reset_index(),
        x="Disks",
        y="Duration (ms)",
        title="Average Access Time",
        color="Disks",
        labels={"Duration (ms)": "Average Access Time (ms)"},
    )

    # Set x-axis tickmode to "array" and provide the index values to only show those
    fig.update_xaxes(tickmode="array", tickvals=df.index)

    # Add unit to hover text using hovertemplate
    fig.update_traces(
        hovertemplate="Disk: %{x}<br>Average Time: %{y:.2f} ms<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data"):
        st.dataframe(df)

    # average by disk and by request type
    df = fdf.groupby(["Disks", "Request"])["Duration (ms)"].mean()

    # Create an interactive grouped bar plot using Plotly
    fig = px.bar(
        df.unstack(),
        color="Request",
        barmode="group",
        title="Average Access Time per Request Type",
        labels={"value": "Average Access Time (ms)"},
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data"):
        st.dataframe(df)


def show_request_size(data: pd.DataFrame):
    # compute average read and write request size in kbytes
    df = (
        (sector_size * data.groupby(["Disks", "Request"])["Length"].mean() / toKB)
        .to_frame()
        .reset_index()
    )

    # rename the columns
    df.rename(columns={"Length": "Avg. Length (KB)"}, inplace=True)

    # Create Plotly bar plot
    fig = px.bar(
        df,
        x="Disks",
        y="Avg. Length (KB)",
        color="Request",
        barmode="group",
        title="Average Request Length (KB)",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data"):
        st.dataframe(df)

    # Define specific "Length" values to count
    length_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4196, 8192]

    # Count occurrences of specific "Length"
    length_counts = (
        data[data["Length"].isin(length_values)]
        .groupby(["Disks", "Request"])["Length"]
        .value_counts()
        .sort_index()
        .to_frame()
        .reset_index()
    )

    for disk_name in length_counts["Disks"].unique():
        df = length_counts[length_counts["Disks"] == disk_name]
        df["Request Length (KB)"] = (df["Length"] * sector_size / toKB).astype(int)
        df.drop(["Disks", "Length"], axis=1, inplace=True)
        # Create Plotly bar plot
        fig = px.bar(
            df,
            x="Request Length (KB)",
            y="count",
            color="Request",
            barmode="group",
            title=f"Request Length Count - {disk_name}",
            custom_data=["Request", "count"],
        )
        fig.update_xaxes(type="category")
        fig.update_traces(
            hovertemplate="Size: %{x}KB<br>Count: %{y}<br>Type: %{customdata[0]}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Show data"):
            st.dataframe(df)


# Define the Streamlit app
def main():
    st.set_page_config(
        page_title="DiskMon Log Analyzer",
        # layout="wide",
        page_icon=":chart_with_upwards_trend:",
        initial_sidebar_state="auto",
    )

    # Set the page title
    st.title(":chart_with_upwards_trend: DiskMon Log Analyzer")

    with st.expander("About DiskMon"):
        st.header("About DiskMon")
        st.write(diskmon)

    with st.expander("About this page"):
        st.header("About this page")
        st.write(summary)

    st.header(":computer: Usage")
    st.write(
        usage.format(
            python_svg=render_svg(python_svg, width="4%"),
            pandas_svg=render_svg(pandas_svg, width="2%"),
        ),
        unsafe_allow_html=True,
    )

    st.header(":outbox_tray: Upload your log file")
    uploaded_file = st.file_uploader(
        "Upload your untouched or compressed DiskMon.LOG file:",
        type=["log", "zip", "gz", "bz2"],
        key="uploaded_file",
    )

    if uploaded_file is None:
        st.warning(
            "You have to upload your log file before we can continue...", icon="⚠️"
        )
        return

    # Read the uploaded log file
    data = read_uploaded_file(uploaded_file)

    # Display a sample of the loaded data
    with st.expander("Sample of the loaded data"):
        st.dataframe(data.head().style.format({"Time": "{:.6}"}))

    # Rename disks if more than one disk is found
    disks_names = sorted(data["Disks"].unique().tolist())
    if len(disks_names) > 1:
        st.header(":pencil: Name your disks")
        st.write(
            "DiskMon reports more than one disk in your system. Rename them below if you want to."
        )
        new_names = {}
        for name in disks_names:
            new_names[name] = st.text_input(name + " name:", name)
        data = update_disks_names(data, new_names)

    # Show log summary
    st.header(":page_facing_up: DiskMon Log Summary")
    st.table(log_summary(data))

    # Plot summary
    plot_summary(data)

    # Show access time info
    st.header(":stopwatch: Access Time")
    with st.expander("Show info"):
        st.write(
            """
            Access time refers to the duration it requires to read or write a specific portion of a file.
            In other words, it is the time taken to complete a request of a particular length.

            The computed access time is not accurate because the duration is not always available in DiskMon log file.
            In that case, the access time is calculated based on the time difference between two consecutive requests,
            which is not accurate because the disk could be in idle state before the next request.
            """
        )
    show_access_time(data)

    st.header(":bar_chart: Request Size")
    with st.expander("Show info"):
        st.write(
            """
            The requested size is the amount of data requested by an application when reading/writing from/to a file.
            It is the product of the number of sectors requested (the Length column) and the sector size (always 512 bytes in DiskMon).

            The most important SSD performance metric for you will depend on your usage pattern.
            The charts below show the average request size and the number of requests for each size.
            This, along with the previously shown read/write and random/sequential ratios,
            can help you determine the most important aspect of an SSD disk to consider for your workload.
            """
        )
    show_request_size(data)


if __name__ == "__main__":
    # Increase pandas default output precision from 6 decimal places to 7
    pd.set_option("display.precision", 7)

    # Set Diskmon column names
    column_names = [
        "Index",
        "Time",
        "Duration (ms)",
        "Disks",
        "Request",
        "Sector",
        "Length",
        "Type",
    ]

    # Define the sector size in bytes
    sector_size = 512  # 512 bytes

    # Default plot width
    plot_width = 800  # False

    # conversion factors from bytes
    toGB = 1024**3
    toMB = 1024**2
    toKB = 1024

    # Run the app
    main()
