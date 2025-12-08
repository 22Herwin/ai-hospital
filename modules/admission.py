"""
Admission Workflow and Timeline Management
"""
import time
import datetime
import pandas as pd
import streamlit as st
from typing import Dict
from .data_manager import (
    load_stock, save_stock, load_hospitals, save_hospitals,
    find_available_hospital, check_ward_capacity_and_alert
)
from .visualizations import (
    create_ward_distribution_chart, create_diagnosis_chart,
    create_severity_pie_chart, create_hospital_ward_heatmap
)


def render_admission_workflow(patient: dict, ai_result: dict):
    """Render admission workflow for inpatients"""
    if not ai_result or not ai_result.get('inpatient'):
        return
    
    st.markdown("---")
    st.subheader("Admission Workflow")

    stock_df = load_stock()
    hospitals_df = load_hospitals()
    recommended_meds = ai_result.get('recommended_medicines', [])
    requested_ward = ai_result.get('ward_type', 'General')

    # Find available hospital
    available_hospital = find_available_hospital(requested_ward, hospitals_df)

    if available_hospital is None:
        st.error(f"No beds available in any hospital for {requested_ward} ward")
        st.info("All hospitals are at full capacity. Consider:")
        st.write("- Placing patient on waiting list")
        st.write("- Transferring to another hospital type")
        st.write("- Delaying non-urgent admission")
        return

    # Show hospital assignment
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Hospital Assignment")
        st.write(f"**Hospital:** {available_hospital['hospital_name']}")
        st.write(f"**Ward Type:** {available_hospital['ward_type']}")
        st.write(f"**Available Beds:** {available_hospital['available_beds']}/{available_hospital['total_beds']}")
    
    with col2:
        st.info(f"**Occupancy:** {int((available_hospital['occupied_beds'] / available_hospital['total_beds']) * 100)}%")

    # Filter available medicines
    available_meds = stock_df[
        stock_df['medicine_name'].apply(lambda x: any(med.lower() in x.lower() for med in recommended_meds)) &
        (stock_df['stock'] > 0)
    ]

    if available_meds.empty:
        st.warning("Exact recommended medicines not in stock. Showing all available medicines:")
        available_meds = stock_df[stock_df['stock'] > 0]
        
        if available_meds.empty:
            st.error("No medicines in stock. Please replenish inventory first.")
            st.stop()

    # Medicine selection
    col1, col2 = st.columns(2)
    with col1:
        selected_med = st.selectbox(
            'Select Medication to Assign',
            available_meds['medicine_name'].tolist(),
            help="Choose from available in-stock medications"
        )

    with col2:
        current_stock = available_meds[available_meds['medicine_name'] == selected_med]['stock'].values[0]
        qty = st.number_input(
            'Quantity to Assign',
            min_value=1,
            max_value=int(current_stock),
            value=1,
            help=f"Available stock: {current_stock}"
        )
    
    # Admission date planning
    st.write("**Planned Admission Schedule**")
    col_date1, col_date2, col_date3 = st.columns(3)
    
    with col_date1:
        admission_date = st.date_input(
            "Admission Date",
            value=datetime.date.today(),
            min_value=datetime.date.today(),
            help="Date when patient will be admitted"
        )
    
    with col_date2:
        admission_time = st.time_input(
            "Admission Time",
            value=datetime.time(hour=9, minute=0),
            help="Time of admission"
        )
    
    with col_date3:
        discharge_days = st.number_input(
            "Estimated Length of Stay (days)",
            min_value=1,
            max_value=90,
            value=ai_result.get('estimated_stay_days', 3),
            help="How many days will patient be admitted?"
        )
    
    # Calculate discharge date
    admission_datetime = datetime.datetime.combine(admission_date, admission_time)
    estimated_discharge = admission_date + datetime.timedelta(days=int(discharge_days))
    
    col_summary1, col_summary2 = st.columns(2)
    with col_summary1:
        st.info(f"📅 **Admission:** {admission_date.strftime('%Y-%m-%d (%A)')} at {admission_time.strftime('%H:%M')}")
    
    with col_summary2:
        st.info(f"📅 **Est. Discharge:** {estimated_discharge.strftime('%Y-%m-%d (%A)')} ({int(discharge_days)} days)")

    # Confirm admission button
    if st.button('CONFIRM ADMISSION', type='primary', use_container_width=True):
        process_admission(
            patient, ai_result, available_hospital,
            selected_med, qty, admission_datetime,
            admission_date, estimated_discharge, discharge_days,
            stock_df, hospitals_df
        )


def process_admission(patient, ai_result, available_hospital, selected_med, qty,
                     admission_datetime, admission_date, estimated_discharge, 
                     discharge_days, stock_df, hospitals_df):
    """Process the admission confirmation"""
    requested_ward = ai_result.get('ward_type', 'General')
    
    # Check capacity
    has_capacity, capacity_msg = check_ward_capacity_and_alert(requested_ward, hospitals_df)
    st.info(capacity_msg)
    
    if not has_capacity:
        st.error(f"Cannot admit: {capacity_msg}")
        st.stop()
    
    # Get diagnosis info with fallback
    diagnosis_code = ai_result.get('icd10_code', 'Unknown')
    diagnosis_name = ai_result.get('diagnosis_name', '')
    
    # If diagnosis_name is missing, try to get it from ICD-10 lookup
    if not diagnosis_name or diagnosis_name == 'Unknown':
        try:
            from icd10_loader import lookup_icd10
            icd_info = lookup_icd10(diagnosis_code)
            if icd_info and icd_info.get('title'):
                diagnosis_name = icd_info['title']
            else:
                diagnosis_name = 'Unknown diagnosis'
        except Exception:
            diagnosis_name = 'Unknown diagnosis'
    
    # Create admission record
    admission_data = {
        'patient_id': patient['pid'],
        'admit_time': admission_datetime.isoformat(),
        'planned_admission_date': admission_date.isoformat(),
        'planned_admission_time': admission_datetime.time().isoformat(),
        'estimated_discharge_date': estimated_discharge.isoformat(),
        'length_of_stay_days': int(discharge_days),
        'hospital_id': available_hospital['hospital_id'],
        'hospital_name': available_hospital['hospital_name'],
        'ward_type': available_hospital['ward_type'],
        'estimated_days': ai_result.get('estimated_stay_days', 3),
        'med_used': selected_med,
        'qty': int(qty),
        'diagnosis_code': diagnosis_code,
        'diagnosis_name': diagnosis_name,
        'severity_score': patient['severity_score']
    }

    # Update hospital occupancy
    hospitals_df.loc[hospitals_df['hospital_id'] == available_hospital['hospital_id'], 'occupied_beds'] = \
        int(available_hospital['occupied_beds']) + 1
    save_hospitals(hospitals_df)

    # Save to SQLite
    sqlite_success = False
    try:
        from sqlite_client import insert_admission
        result = insert_admission(admission_data)
        if result:
            sqlite_success = True
            st.success(f"Patient {patient['pid']} admitted to {admission_data['hospital_name']} ({admission_data['ward_type']} Ward)")
        else:
            st.error("Failed to save to SQLite")
            st.stop()
    except ImportError:
        st.error("SQLite client not available")
        st.stop()
    except Exception as e:
        st.error(f"SQLite error: {type(e).__name__}: {str(e)}")
        st.stop()

    # Update stock
    stock_df.loc[stock_df['medicine_name'] == selected_med, 'stock'] -= qty
    save_stock(stock_df)

    # Success messages
    st.success(f"Patient {patient['pid']} admitted")
    st.info(f"Medicine assigned: {selected_med} x{qty}")
    st.info(f"Admission: {admission_date.strftime('%Y-%m-%d')} → Discharge: {estimated_discharge.strftime('%Y-%m-%d')} ({int(discharge_days)} days)")
    st.info(f"Data stored in SQLite")
    
    # Check occupancy after admission
    updated_hospitals_df = load_hospitals()
    hospital_data = updated_hospitals_df[updated_hospitals_df['hospital_id'] == available_hospital['hospital_id']]
    if not hospital_data.empty:
        updated_total = int(hospital_data.iloc[0]['total_beds'])
        updated_occupied = int(hospital_data.iloc[0]['occupied_beds'])
        occupancy_pct = int((updated_occupied / updated_total) * 100) if updated_total > 0 else 0
        
        if occupancy_pct >= 85:
            st.warning(f"Ward now at {occupancy_pct}% capacity")
    
    st.session_state.last_admission_id = patient['pid']
    st.session_state.admission_complete = True
    
    time.sleep(2)
    
    # Clear caches
    load_hospitals.clear()
    load_stock.clear()
    st.rerun()


def render_admission_timeline():
    """Render patient admission timeline with filtering and analytics"""
    st.markdown("---")
    st.subheader("Patient Admission Timeline")

    try:
        from sqlite_client import get_all_admissions
        
        # Only fetch recent admissions by default (limit 50 most recent)
        admissions = get_all_admissions()
        
        if not admissions or len(admissions) == 0:
            st.info("No admission data yet. Admissions will appear here as patients are admitted.")
            return
        
        # Limit to most recent 50 for performance
        admissions = admissions[:50]
        
        admissions_df = pd.DataFrame(admissions)
        
        # Convert datetime with multiple format handling
        admissions_df['admit_time'] = pd.to_datetime(admissions_df['admit_time'], errors='coerce', utc=True)
        
        # Drop rows with invalid dates (can't be parsed)
        valid_dates_mask = admissions_df['admit_time'].notna()
        if not valid_dates_mask.all():
            invalid_count = (~valid_dates_mask).sum()
            if invalid_count > 0:
                st.warning(f"⚠️ {invalid_count} record(s) with invalid dates were filtered out")
            admissions_df = admissions_df[valid_dates_mask].copy()
        
        if len(admissions_df) == 0:
            st.info("No valid admission records found. Please check your database.")
            return
        
        admissions_df = admissions_df.sort_values('admit_time', ascending=False)
        
        # Extract date components
        admissions_df['date'] = admissions_df['admit_time'].dt.date
        admissions_df['month'] = admissions_df['admit_time'].dt.to_period('M')
        admissions_df['year'] = admissions_df['admit_time'].dt.year
        
        # Fill empty diagnosis names with 'Not specified'
        admissions_df['diagnosis_name'] = admissions_df['diagnosis_name'].fillna('Not specified')
        admissions_df['diagnosis_name'] = admissions_df['diagnosis_name'].replace('', 'Not specified')
        
        # Create tabs
        timeline_tab1, timeline_tab2, timeline_tab3 = st.tabs([
            "Recent Admissions", "Analytics", "Hospital Breakdown"
        ])
        
        with timeline_tab1:
            render_recent_admissions_tab(admissions_df)
        
        with timeline_tab2:
            render_analytics_tab(admissions_df)
        
        with timeline_tab3:
            render_hospital_breakdown_tab(admissions_df)

    except ImportError:
        st.warning("SQLite client not configured. Timeline feature unavailable.")
    except Exception as e:
        st.error(f"Error loading timeline: {str(e)}")


def render_recent_admissions_tab(admissions_df: pd.DataFrame):
    """Render recent admissions tab with filtering"""
    st.write("**Filter Options**")
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        filter_type = st.radio(
            "Filter by:",
            ["All", "Date Range", "Specific Month", "Specific Date"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    filtered_df = admissions_df.copy()
    
    with col_filter2:
        if filter_type == "Date Range":
            date_range = st.date_input(
                "Select date range",
                value=(admissions_df['date'].min(), admissions_df['date'].max()),
                min_value=admissions_df['date'].min(),
                max_value=admissions_df['date'].max(),
                key="date_range"
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
        
        elif filter_type == "Specific Month":
            available_months = sorted(admissions_df['month'].unique(), reverse=True)
            selected_month = st.selectbox(
                "Select month",
                available_months,
                format_func=lambda x: str(x),
                key="month_filter"
            )
            filtered_df = filtered_df[filtered_df['month'] == selected_month]
        
        elif filter_type == "Specific Date":
            available_dates = sorted(admissions_df['date'].unique(), reverse=True)
            selected_date = st.selectbox(
                "Select date",
                available_dates,
                format_func=lambda x: x.strftime('%Y-%m-%d'),
                key="date_filter"
            )
            filtered_df = filtered_df[filtered_df['date'] == selected_date]
    
    with col_filter3:
        filter_ward = st.multiselect(
            "Filter by Ward Type",
            admissions_df['ward_type'].unique().tolist(),
            default=admissions_df['ward_type'].unique().tolist(),
            key="ward_filter"
        )
        if filter_ward:
            filtered_df = filtered_df[filtered_df['ward_type'].isin(filter_ward)]
    
    st.divider()
    
    # Limit display for performance
    display_limit = 20
    total_filtered = len(filtered_df)
    
    st.write(f"**Showing {min(display_limit, total_filtered)} of {total_filtered} admission(s)** (from {len(admissions_df)} recent records)")
    
    if len(filtered_df) > 0:
        # Limit to top 20 for performance
        display_filtered_df = filtered_df.head(display_limit)
        
        # Table view
        recent_admissions = display_filtered_df.head(10).copy()
        # Handle NaT values safely
        recent_admissions['admit_time'] = recent_admissions['admit_time'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else 'N/A'
        )
        
        display_df = recent_admissions[[
            'patient_id', 'admit_time', 'ward_type', 
            'diagnosis_code', 'med_used', 'severity_score'
        ]].copy()
        
        display_df.columns = ['Patient ID', 'Admit Time', 'Ward', 'Diagnosis', 'Medication', 'Severity']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Expandable detailed view (limit to 20 for performance)
        for idx, row in display_filtered_df.iterrows():
            # Safely format dates
            admit_time_short = row['admit_time'].strftime('%m-%d %H:%M') if pd.notna(row['admit_time']) else 'N/A'
            admit_time_full = row['admit_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['admit_time']) else 'N/A'
            
            with st.expander(f"{row['patient_id']} | {admit_time_short} | {row['severity_score']}/25", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Patient ID:** {row['patient_id']}")
                    st.write(f"**Admission Time:** {admit_time_full}")
                with col2:
                    st.write(f"**Ward Type:** {row['ward_type']}")
                with col3:
                    st.write(f"**Diagnosis Code:** {row['diagnosis_code']}")
                    # Use stored diagnosis_name without expensive ICD-10 lookup
                    diagnosis_name = row.get('diagnosis_name', 'Not specified')
                    if pd.isna(diagnosis_name) or diagnosis_name == '':
                        diagnosis_name = 'Not specified'
                    st.write(f"**Diagnosis Name:** {diagnosis_name}")
                
                st.divider()
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.write(f"**Medication:** {row['med_used']}")
                with col5:
                    st.write(f"**Quantity:** {row['qty']} unit(s)")
                with col6:
                    st.write(f"**Severity Score:** {row['severity_score']}/25")
                
                st.write(f"**Estimated Stay:** {row['estimated_days']} days")
    else:
        st.info("No admissions found for the selected filter(s)")


def render_analytics_tab(admissions_df: pd.DataFrame):
    """Render analytics tab with metrics and charts"""
    col_filter_a1, col_filter_a2 = st.columns(2)
    
    with col_filter_a1:
        filter_type_analytics = st.radio(
            "Analytics Filter:",
            ["All Time", "Date Range", "Specific Month"],
            horizontal=True,
            label_visibility="collapsed",
            key="analytics_filter"
        )
    
    analytics_df = admissions_df.copy()
    
    with col_filter_a2:
        if filter_type_analytics == "Date Range":
            date_range_analytics = st.date_input(
                "Select date range for analytics",
                value=(admissions_df['date'].min(), admissions_df['date'].max()),
                min_value=admissions_df['date'].min(),
                max_value=admissions_df['date'].max(),
                key="date_range_analytics"
            )
            if isinstance(date_range_analytics, tuple) and len(date_range_analytics) == 2:
                start_date_a, end_date_a = date_range_analytics
                analytics_df = analytics_df[(analytics_df['date'] >= start_date_a) & (analytics_df['date'] <= end_date_a)]
        
        elif filter_type_analytics == "Specific Month":
            available_months_a = sorted(admissions_df['month'].unique(), reverse=True)
            selected_month_a = st.selectbox(
                "Select month for analytics",
                available_months_a,
                format_func=lambda x: str(x),
                key="month_filter_analytics"
            )
            analytics_df = analytics_df[analytics_df['month'] == selected_month_a]
    
    st.divider()
    st.write("**Key Metrics**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Admissions", len(analytics_df), delta="records")
    
    with col2:
        avg_severity = analytics_df['severity_score'].mean() if len(analytics_df) > 0 else 0
        severity_label = "Critical" if avg_severity >= 15 else "High" if avg_severity >= 10 else "Moderate" if avg_severity >= 5 else "Low"
        st.metric("Avg Severity", f"{avg_severity:.1f}/25", delta=severity_label)
    
    with col3:
        avg_stay = analytics_df['estimated_days'].mean() if len(analytics_df) > 0 else 0
        st.metric("Avg Stay (days)", f"{avg_stay:.1f}", delta="inpatient only")
    
    with col4:
        inpatient_count = len(analytics_df[analytics_df['estimated_days'] > 0])
        inpatient_pct = int((inpatient_count / len(analytics_df) * 100)) if len(analytics_df) > 0 else 0
        st.metric("Inpatient Rate", f"{inpatient_pct}%", delta=f"{inpatient_count} patients")
    
    st.divider()
    
    if len(analytics_df) > 0:
        # Charts
        ward_count = analytics_df['ward_type'].value_counts()
        st.plotly_chart(create_ward_distribution_chart(ward_count), use_container_width=True)
        
        diagnosis_count = analytics_df['diagnosis_code'].value_counts().head(8)
        st.plotly_chart(create_diagnosis_chart(diagnosis_count), use_container_width=True)
        
        severity_bins = pd.cut(analytics_df['severity_score'], bins=[0, 5, 10, 15, 25], 
                              labels=['Low (0-5)', 'Moderate (5-10)', 'High (10-15)', 'Critical (15+)'])
        severity_count = severity_bins.value_counts().sort_index()
        st.plotly_chart(create_severity_pie_chart(severity_count), use_container_width=True)
    else:
        st.info("No data available for the selected filter")


def render_hospital_breakdown_tab(admissions_df: pd.DataFrame):
    """Render hospital breakdown tab"""
    col_hosp1, col_hosp2 = st.columns(2)
    
    with col_hosp1:
        hospital_filter = st.multiselect(
            "Filter by Hospital",
            admissions_df['hospital_name'].unique().tolist(),
            default=admissions_df['hospital_name'].unique().tolist(),
            key="hospital_filter"
        )
    
    hospital_df = admissions_df[admissions_df['hospital_name'].isin(hospital_filter)]
    
    st.write("**Hospital Statistics**")
    
    hospital_stats = hospital_df.groupby('hospital_name').agg({
        'patient_id': 'count',
        'severity_score': 'mean',
        'estimated_days': 'mean'
    }).round(2)
    
    hospital_stats.columns = ['Total Admits', 'Avg Severity', 'Avg Stay (days)']
    hospital_stats = hospital_stats.sort_values('Total Admits', ascending=False)
    
    st.dataframe(hospital_stats, use_container_width=True)
    
    st.divider()
    
    if len(hospital_df) > 0:
        hospital_ward = pd.crosstab(hospital_df['hospital_name'], hospital_df['ward_type'])
        st.plotly_chart(create_hospital_ward_heatmap(hospital_ward), use_container_width=True)
        st.dataframe(hospital_ward, use_container_width=True)
