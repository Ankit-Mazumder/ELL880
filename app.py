from flask import Flask, render_template, request,jsonify,redirect
import sqlite3
from datetime import datetime
import re
import plotly
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.offline as pyo
import io
import ast
from flask import send_file
import math
from database_rp import create_cache1,insert_cache1,create_cache2,insert_cache2,create_cache3,insert_cache3,create_cache4,insert_cache4,create_cache5,insert_cache5,create_cache6,insert_cache6,create_cache7,insert_cache7,create_cache8,insert_cache8
import ast

app = Flask(__name__)


def clean_string(s: str) -> str:
    s = re.sub('[^a-zA-Z0-9\s]', ' ', s)
    s = re.sub('[^\S\n]+', ' ', s)
    s = s.strip()
    return s




@app.route('/')
def index():
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT code from conferences")
    rows = cursor.fetchall()
    optns = [row[0] for row in rows]
    current_year = datetime.now().year
    return render_template('welcome.html', options=optns, current_year=current_year)


def yearsCitations(paperId):
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT year,citation_count FROM table2 WHERE paper_id = ?",(paperId,))
    l = cursor.fetchall()
    y = [item[0] for item in l]
    c = [item[1] for item in l]
    return y,c


@app.route('/submit', methods=['GET'])
def submit_query():
    code = request.args.get('q')
    year = int(request.args.get('year'))
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    code = clean_string(str(code))
    cursor.execute("""
    SELECT paperId,title, citationCount 
    FROM rp_data 
    WHERE publication = ? 
        AND year >= ? 
    ORDER BY citationCount DESC  
    LIMIT (SELECT ROUND(COUNT(*) * 0.05) 
           FROM rp_data 
           WHERE publication = ?
               AND year >= ?)
    """,(code, int(year), code, int(year)))

    results = cursor.fetchall()
    return jsonify(results)
    # return jsonify({'success': True})


papers_info = []
@app.route('/paperdetails', methods=['GET'])
def graph_query():
    paperId = request.args.get('q')
    for item in papers_info:
        if item["paper"] == paperId: 
            print('cache')
            return jsonify(item["results"])
    p = []
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM rp_data WHERE paperId = ?",(paperId,))
    query = cursor.fetchall()
    y,c  = yearsCitations(paperId)

    cursor1 = conn.cursor()
    cursor1.execute("select abstract FROM superSetAnalysis WHERE paperId = ?",(paperId,))
    rows = cursor1.fetchall()
    p.extend([query,rows, y, c])
    query_dict = {'paper' : paperId,'results' : p}
    papers_info.append(query_dict)
    print('linees',p)
    return jsonify(p)


timeline_five = []      
@app.route('/timeline', methods=['GET'])
def timeline_query():
    code = request.args.get('q')
    year = int(request.args.get('year'))
    for item in timeline_five:
        if item['code'] == code and item['year'] == year:
            print('cache')
            return jsonify(item["results"])
    ra = []
    print(year,code)
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT paperId,title,citationCount FROM rp_data WHERE publication = ? AND year = ? ORDER BY citationCount DESC  LIMIT (SELECT ROUND(COUNT(*) * 0.10) FROM rp_data WHERE publication = ? AND year = ?) ",(code,year,code,year,))
    resdf = cursor.fetchall()
    cursor1 = conn.cursor()
    cursor1.execute("SELECT paperId,title,citationCount FROM rp_data WHERE publication = ? AND year = ? ORDER BY citationCount ASC LIMIT (SELECT ROUND(COUNT(*) * 0.10) FROM rp_data WHERE publication = ? AND year = ?) ",(code,year,code,year,))
    resaf = cursor1.fetchall() 
    ra.extend([resdf,resaf])
    query_dict = {'code' : code,'year': year,'results' : ra}
    timeline_five.append(query_dict)
    print(ra)
    return jsonify(ra)



@app.route('/search')
def search():
    query = request.args.get('q')
    if not query:
        return jsonify([])
    else:
        conn = sqlite3.connect('complete_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM rp_data WHERE title LIKE ? ", ('%' + query + '%',))
        results = cursor.fetchall()
        matching_strings = [result[0] for result in results]
        last_ten_matching_strings = matching_strings[:20]
        return jsonify(last_ten_matching_strings)


def retrieve_results_query(selected_paper):
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rp_data WHERE title LIKE ?", ('%' + selected_paper + '%',))
    query = cursor.fetchall()
    query = query[:300]
    return query

def retrieve_results_query_conf(selected_paper,conference):
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rp_data WHERE title LIKE ? AND publication = ?", ('%' + selected_paper + '%',conference,))
    query = cursor.fetchall()
    query = query[:300]
    return query




#citation button
@app.route('/paper', methods=['GET'])
def sort_papers():
    selected_paper = request.args.get('q')
    page = int(request.args.get('page', 1))
    results_per_page = int(request.args.get('results_per_page', 20))
    conference = request.args.get('conference')
    if not selected_paper:
        return jsonify([])
    if not conference:
        try:
            conn = sqlite3.connect('complete_data.db')
            cursor = conn.cursor()
            print("error entered")
            cursor.execute("SELECT results,total_pages FROM cache1 WHERE selected_paper = ? ", (selected_paper,))
            r = cursor.fetchall()
            print(r)
            if len(r) > 0:
                results = r[0][0]
                results = ast.literal_eval(results)
                total_pages = r[0][1]
                rc = results
                total_pages = total_pages
                start_index = (page - 1) * results_per_page
                end_index = start_index + results_per_page
                paginated_results = rc[start_index:end_index]
                # print(paginated_results)
                list_c = []
                for item in paginated_results:
                    t = []
                    y, c = yearsCitations(item[0])
                    t.extend([item, y, c])
                    list_c.append(t)

                # print(list_c,len(list_c))
                return jsonify({
                    'results': list_c,
                    'totalPages': total_pages
                })
        except Exception as e:
            print("An exception occurred:", e)
        citations_c = []
        conn = sqlite3.connect('complete_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rp_data WHERE title LIKE ? ORDER BY citationCount DESC LIMIT 300", ('%' + selected_paper + '%',))
        query = cursor.fetchall()
        total_pages = int(len(query) / 20)
        s_query = str(query)
        query_dict = {'selected_paper' : selected_paper,'results' : s_query,'total_pages': total_pages}
        citations_c.append(query_dict)
        create_cache1()
        insert_cache1(citations_c)
        print("came out")
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = query[start_index:end_index]

        list_c = []
        for item in paginated_results:
            t = []
            y, c = yearsCitations(item[0])
            t.extend([item, y, c])
            list_c.append(t)

            # print(list_c,len(list_c))
        return jsonify({
            'results': list_c,
            'totalPages': total_pages
        })
    else:
        try:
            conn = sqlite3.connect('complete_data.db')
            cursor = conn.cursor()
            print("error entered")
            cursor.execute("SELECT results,total_pages FROM cache2 WHERE selected_paper = ? and conference = ? ", (selected_paper,conference,))
            r = cursor.fetchall()
            print(r)
            if len(r) > 0:
                results = r[0][0]
                results = ast.literal_eval(results)
                total_pages = r[0][1]
                rc = results
                total_pages = total_pages
                start_index = (page - 1) * results_per_page
                end_index = start_index + results_per_page
                paginated_results = rc[start_index:end_index]

                list_c = []
                for item in paginated_results:
                    t = []
                    y, c = yearsCitations(item[0])
                    t.extend([item, y, c])
                    list_c.append(t)

                    # print(list_c,len(list_c))
                return jsonify({
                    'results': list_c,
                    'totalPages': total_pages
                })
        except Exception as e:
            print("An exception occurred:", e)
        citations_c = []
        conn = sqlite3.connect('complete_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rp_data WHERE title LIKE ? AND publication = ? ORDER BY citationCount DESC LIMIT 300", ('%' + selected_paper + '%',conference))
        query = cursor.fetchall()
        total_pages = int(len(query) / 20)
        s_query = str(query)
        query_dict = {'selected_paper' : selected_paper,'results' : s_query,'total_pages': total_pages,'conference': conference}
        citations_c.append(query_dict)
        create_cache2()
        insert_cache2(citations_c)
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = query[start_index:end_index]

        list_c = []
        for item in paginated_results:
            t = []
            y, c = yearsCitations(item[0])
            t.extend([item, y, c])
            list_c.append(t)

            # print(list_c,len(list_c))
        return jsonify({
            'results': list_c,
            'totalPages': total_pages
        })




#year button
@app.route('/papery', methods=['GET'])
def sort_papers_y():
    selected_paper = request.args.get('q')
    page = int(request.args.get('page', 1))
    results_per_page = int(request.args.get('results_per_page', 20))
    conference = request.args.get('conference')
    if not selected_paper:
        return jsonify([])
    if not conference:
        try:
            conn = sqlite3.connect('complete_data.db')
            cursor = conn.cursor()
            print("error entered")
            cursor.execute("SELECT results,total_pages FROM cache3 WHERE selected_paper = ? ", (selected_paper,))
            r = cursor.fetchall()
            print(r)
            if len(r) > 0:
                results = r[0][0]
                results = ast.literal_eval(results)
                total_pages = r[0][1]
                rc = results
                total_pages = total_pages
                start_index = (page - 1) * results_per_page
                end_index = start_index + results_per_page
                paginated_results = rc[start_index:end_index]
                list_c = []
                for item in paginated_results:
                    t = []
                    y, c = yearsCitations(item[0])
                    t.extend([item, y, c])
                    list_c.append(t)

                # print(list_c,len(list_c))
                return jsonify({
                    'results': list_c,
                    'totalPages': total_pages
                })
        except Exception as e:
            print("An exception occurred:", e)
        years_c =[]
        print('entered_paper')
        conn = sqlite3.connect('complete_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rp_data WHERE title LIKE ? ORDER BY year DESC LIMIT 300", ('%' + selected_paper + '%',))
        query = cursor.fetchall()
        # print(query,len(query))
        total_pages = int(len(query) / 20)
        s_query = str(query)
        query_dict = {'selected_paper' : selected_paper,'results' : s_query,'total_pages': total_pages}
        years_c.append(query_dict)
        create_cache3()
        insert_cache3(years_c)
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = query[start_index:end_index]
        # print('emotional',query[240:260])

        list_c = []
        for item in paginated_results:
            t = []
            y, c = yearsCitations(item[0])
            t.extend([item, y, c])
            list_c.append(t)

            # print(list_c,len(list_c))
        return jsonify({
            'results': list_c,
            'totalPages': total_pages
        })
    else:
        try:
            conn = sqlite3.connect('complete_data.db')
            cursor = conn.cursor()
            print("error entered")
            cursor.execute("SELECT results,total_pages FROM cache4 WHERE selected_paper = ? and conference = ? ", (selected_paper,conference,))
            r = cursor.fetchall()
            print(r)
            if len(r) > 0:
                results = r[0][0]
                results = ast.literal_eval(results)
                total_pages = r[0][1]
                rc = results
                total_pages = total_pages
                start_index = (page - 1) * results_per_page
                end_index = start_index + results_per_page
                paginated_results = rc[start_index:end_index]
                list_c = []
                for item in paginated_results:
                    t = []
                    y, c = yearsCitations(item[0])
                    t.extend([item, y, c])
                    list_c.append(t)

                # print(list_c,len(list_c))
                print(list_c,'enteredCache')
                return jsonify({
                    'results': list_c,
                    'totalPages': total_pages
                })
        except Exception as e:
            print("An exception occurred:", e)
        years_c = []
        print('entered_paper')
        conn = sqlite3.connect('complete_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rp_data WHERE title LIKE ? AND publication = ? ORDER BY year DESC LIMIT 300", ('%' + selected_paper + '%',conference,))
        query = cursor.fetchall()
        # print(query,len(query))
        total_pages = int(len(query) / 20)
        s_query = str(query)
        query_dict = {'selected_paper' : selected_paper,'results' : s_query,'total_pages': total_pages,'conference': conference}
        years_c.append(query_dict)
        create_cache4()
        insert_cache4(years_c)
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = query[start_index:end_index]
        # print('emotional',query[240:260])

        list_c = []
        for item in paginated_results:
            t = []
            y, c = yearsCitations(item[0])
            t.extend([item, y, c])
            list_c.append(t)

            # print(list_c,len(list_c))
        return jsonify({
            'results': list_c,
            'totalPages': total_pages
        })
        

        

@app.route('/cited',methods = ['GET','POST'] )
def cited_paper():
    print('entered_cited')
    paper = request.json.get('q')
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT citations from superSetAnalysis WHERE paperId = ?",(paper,))
    r = cursor.fetchall()
    list_cited = []
    for i in range(len(r)):
        t = []
        l = str(r)
        cl = l[3:-4]
        ld = ast.literal_eval(cl)
        a = ld[0]['authors']
        au =''
        for i in range(len(a)):
            au = au + a[i]['name'] + ','
        name = ld[0]['title']
        year = ld[0]['year']
        venue = ld[0]['venue']
        isInf = ld[0]['isInfluential']
        t.extend([name,au,year,venue,isInf])
        list_cited.append(t)
    return jsonify(list_cited)
    





def normalize_last_values(data):
    max_value = max([sublist[-1] for sublist in data]) 
    min_value = min([sublist[-1] for sublist in data]) 
    if(min_value == max_value):
        for sublist in data:
            last_value = sublist[-1]
            sublist.append(last_value)  
    
    else:
        for sublist in data:
            last_value = sublist[-1]  
            normalized_value = round(((last_value - min_value) / (max_value - min_value)) * 100,2)
            sublist.append(normalized_value)  
        
    return data
    





#significant button
def weighted_approach(item,y,c):
    alpha =0.6
    t = []
    if len(c) == 1:
        avg1 = round(alpha*c[0],2)
        t.extend([item, y, c, avg1])
        return t
    elif len(c) == 2:
        avg1 = round(alpha*c[0] + pow(alpha,2) * c[1],2)
        t.extend([item, y, c, avg1])
        return t
    elif len(c) == 3:
        avg1 = round(alpha * c[0] + pow(alpha,2) * c[1] +  pow(alpha,3)*c[2],2)
        t.extend([item, y, c, avg1])
        return t
    elif len(c) == 4:
        avg1 = round(alpha * c[0] +  pow(alpha,2) * c[1] + pow(alpha,3) * c[2] +  pow(alpha,4)*c[3],2)
        t.extend([item, y, c, avg1])
        return t
    elif len(c) == 5:
        avg1 = round(alpha * c[0] +  pow(alpha,2) * c[1] +  pow(alpha,3) * c[2] +  pow(alpha,4) * c[3] +  pow(alpha,5) * c[4],2)
        t.extend([item, y, c, avg1])
        return t

def cache5(selected_paper,filter_option,page):
    results_per_page = 20
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    print("error entered")
    cursor.execute("SELECT results,total_pages FROM cache5 WHERE selected_paper = ?  and algorithm = ?", (selected_paper,filter_option))
    r = cursor.fetchall()
    print(r)
    if len(r) > 0:
        results = r[0][0]
        results = ast.literal_eval(results)
        total_pages = r[0][1]
        rc = results
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = rc[start_index:end_index]
        return paginated_results,total_pages

def item_per_query(item):
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    pap = clean_string(str(item[0]))
    current_year = datetime.now().year
    start_year = current_year - 5
    query1 = "SELECT year, citation_count FROM table2 WHERE year >= ? AND year <= ? AND paper_id = ?"
    cursor.execute(query1, (start_year, current_year, pap))
    q_r = cursor.fetchall()  
    y = [ite[0] for ite in q_r]
    c = [ite[1] for ite in q_r]
    return y,c

def database_entry_cache5(page,selected_paper,algorithm,sorted_h,total_pages):
    sig_h = []
    results_per_page = 20
    s_sorted_h = str(sorted_h)
    query_dict = {'selected_paper' : selected_paper,'results' : s_sorted_h,'total_pages': total_pages,'algorithm': algorithm}
    sig_h.append(query_dict)
    create_cache5()
    insert_cache5(sig_h)
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    paginated_results = sorted_h[start_index:end_index]
    return paginated_results,total_pages

def cache6(selected_paper,filter_option,page,conference):
    results_per_page = 20
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    print("error entered")
    cursor.execute("SELECT results,total_pages FROM cache6 WHERE selected_paper = ? and conference = ? and algorithm = ?", (selected_paper,conference,filter_option,))
    r = cursor.fetchall()
    if len(r) > 0:
        results = r[0][0]
        results = ast.literal_eval(results)
        total_pages = r[0][1]
        rc = results
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = rc[start_index:end_index]
        return paginated_results,total_pages 

def database_entry_cache6(page,selected_paper,algorithm,sorted_h,total_pages,conference):
    results_per_page = 20
    sig_h = []
    s_sorted_h = str(sorted_h)
    query_dict = {'selected_paper' : selected_paper,'results' : s_sorted_h,'total_pages': total_pages,'conference': conference,'algorithm': algorithm}
    sig_h.append(query_dict)
    create_cache6()
    insert_cache6(sig_h)
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    paginated_results = sorted_h[start_index:end_index]
    return paginated_results,total_pages

def incremental_increase(item):
    paper = clean_string(str(item[0]))
    venue = clean_string(str(item[5]))
    # cursor.execute("SELECT count(paper_id) from cited_papers WHERE paper_id = ?",(paper,))
    # r = cursor.fetchall()
    # num = r[0][0]
    conn = sqlite3.connect('complete_data.db')
    cursor1 = conn.cursor()
    cursor2 = conn.cursor()
    current_year = datetime.now().year
    start_year = current_year - 5
    query1 = 'SELECT count(paperId) FROM rp_data WHERE year >= ? AND year <= ? AND publication = ?' #error in calculating formula: (no of citations from these 50 conferences/total no of papers published in 50 conferences in that period)
    cursor1.execute(query1, (start_year, current_year, venue))
    d = cursor1.fetchall()
    denom = d[0][0]
    query2 = "SELECT year, citation_count FROM table2 WHERE year >= ? AND year <= ? AND paper_id = ?"
    cursor2.execute(query2, (start_year, current_year, paper))
    q_r = cursor2.fetchall()
    y = [ite[0] for ite in q_r]
    c = [ite[1] for ite in q_r]
    return y,c,denom


@app.route('/paperh', methods=['GET'])
def sort_papers_h():
    selected_paper = request.args.get('q')
    filter_option = request.args.get('filter')
    page = int(request.args.get('page', 1))
    results_per_page = int(request.args.get('results_per_page', 20))
    conference = request.args.get('conference')
    if not selected_paper:
        return jsonify([])
    else:
        if not conference:
            if filter_option == 'weighted_approach':
                try:
                    paginated_results,total_pages = cache5(selected_paper,filter_option,page)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                list_h = []
                query = retrieve_results_query(selected_paper)
                for item in query:
                    t = []
                    y,c = item_per_query(item)
                    if len(c) == 0:
                        continue
                    t = weighted_approach(item,y,c)
                    list_h.append(t)
                sorted_h = sorted(list_h, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache5(page,selected_paper,filter_option,sorted_h,total_pages)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
        
            elif filter_option == 'simple_average':
                try:
                    paginated_results,total_pages = cache5(selected_paper,filter_option,page)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                list_h = []
                query = retrieve_results_query(selected_paper)
                for item in query:
                    t = []
                    y,c = item_per_query(item)
                    if len(c) == 0:
                        continue
                    avg1 = round(sum(c) / len(c),2)
                    t.extend([item, y, c, avg1])
                    list_h.append(t)
                sorted_h = sorted(list_h, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache5(page,selected_paper,filter_option,sorted_h,total_pages)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
                
            elif filter_option == 'incremental_increase':
                try:
                    paginated_results,total_pages = cache5(selected_paper,filter_option,page)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                sig_h = []
                query = retrieve_results_query(selected_paper)
                list_h = []
                for item in query:
                    t = []
                    y,c,denom = incremental_increase(item)
                    if len(c) == 0 or denom == 0:
                        continue
                    else:
                        cI = round(sum(c)/denom,2)
                        t.extend([item, y, c, cI])
                        list_h.append(t)
                # list_h = normalize_last_values(list_h)
                sorted_h = sorted(list_h, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache5(page,selected_paper,filter_option,sorted_h,total_pages)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
        else:
            if filter_option == 'weighted_approach':
                try:
                    paginated_results,total_pages = cache6(selected_paper,filter_option,page,conference)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                query = retrieve_results_query_conf(selected_paper,conference)
                list_h = []
                for item in query:
                    t = []
                    y,c = item_per_query(item)
                    if len(c) == 0:
                        continue
                    t = weighted_approach(item,y,c)
                    list_h.append(t)
                sorted_h = sorted(list_h, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache6(page,selected_paper,filter_option,sorted_h,total_pages,conference)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
        
            elif filter_option == 'simple_average':
                try:
                    paginated_results,total_pages = cache6(selected_paper,filter_option,page,conference)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })  
                except Exception as e:
                    print("An exception occurred:", e)
                query = retrieve_results_query_conf(selected_paper,conference)
                list_h=[]
                for item in query:
                    t = []
                    y,c = item_per_query(item)
                    if len(c) == 0:
                        continue
                    avg1 = round(sum(c) / len(c),2)
                    t.extend([item, y, c, avg1])
                    list_h.append(t)
                sorted_h = sorted(list_h, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache6(page,selected_paper,filter_option,sorted_h,total_pages,conference)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
                
            elif filter_option == 'incremental_increase':
                try:
                    paginated_results,total_pages = cache6(selected_paper,filter_option,page,conference)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                query = retrieve_results_query_conf(selected_paper,conference)
                list_h = []
                for item in query:
                    t = []
                    y,c,denom = incremental_increase(item)
                    if len(c) == 0 or denom == 0:
                        continue
                    else:
                        cI = round(sum(c)/denom,2)
                        t.extend([item, y, c, cI])
                        list_h.append(t)
                # list_h = normalize_last_values(list_h)
                sorted_h = sorted(list_h, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache6(page,selected_paper,filter_option,sorted_h,total_pages,conference)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })






            




def scoring(arr,alpha):
    positions = [i for i in range(len(arr))]
    positions.sort(key=lambda a: arr[a])
    scores = [0 for i in range(len(arr))]
    for i in range(len(positions)):
        scores[positions[i]] = round(float(arr[positions[i]]) * float((i + 1)) * alpha,2)
    return scores

def periodwise(years, citations, period,alpha):
    year_span = ["-".join(map(str, years[i:i+period])) for i in range(len(years)-period+1)]
    citations_span = [sum(citations[i:i+period]) for i in range(len(citations)-period+1)]
    citations_scores = scoring(citations_span,alpha)
    return list(zip(year_span, citations_scores))

def bestTimePeriod(y,c,alpha):
    v = []
    for i in range(len(c)):
        l = periodwise(y, c, i+1,alpha)
        v.append(l)
    data = [item for sublist in v for item in sublist]
    sorted_list = sorted(data, key=lambda x: x[1], reverse=True)
    cI = sorted_list[0][1]
    period = sorted_list[0][0]
    return cI,period

def bestTimePeriodDynamic(y,c,alpha):
    v = []
    for i in range(len(c)):
        l = periodwise(y, c, i+1,alpha)
        v.append(l)
        alpha = alpha*alpha
    data = [item for sublist in v for item in sublist]
    sorted_list = sorted(data, key=lambda x: x[1], reverse=True)
    cI = sorted_list[0][1]
    period = sorted_list[0][0]
    return cI,period



#best button
def cache7(selected_paper,filter_option,page):
    results_per_page = 20
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    print("error entered")
    cursor.execute("SELECT results,total_pages FROM cache7 WHERE selected_paper = ?  and algorithm = ?", (selected_paper,filter_option))
    r = cursor.fetchall()
    if len(r) > 0:
        results = r[0][0]
        results = ast.literal_eval(results)
        total_pages = r[0][1]
        rc = results
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = rc[start_index:end_index]
        return paginated_results,total_pages

def database_entry_cache7(page,selected_paper,algorithm,sorted_b,total_pages):
    results_per_page = 20
    sig_b = []
    s_sorted_b = str(sorted_b)
    query_dict = {'selected_paper' : selected_paper,'results' :s_sorted_b,'total_pages': total_pages,'algorithm': algorithm}
    sig_b.append(query_dict)
    create_cache7()
    insert_cache7(sig_b)
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    paginated_results = sorted_b[start_index:end_index]
    return paginated_results,total_pages

def cache8(selected_paper,filter_option,page,conference):
    results_per_page = 20
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    print("error entered")
    cursor.execute("SELECT results,total_pages FROM cache8 WHERE selected_paper = ? and conference = ? and algorithm = ?", (selected_paper,conference,filter_option,))
    r = cursor.fetchall()
    if len(r) > 0:
        results = r[0][0]
        results = ast.literal_eval(results)
        total_pages = r[0][1]
        rc = results
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = rc[start_index:end_index]
        return paginated_results,total_pages

def database_entry_cache8(page,selected_paper,algorithm,sorted_b,total_pages,conference):
    sig_b = []
    results_per_page = 20
    s_sorted_b = str(sorted_b)
    query_dict = {'selected_paper' : selected_paper,'results' :s_sorted_b,'total_pages': total_pages,'algorithm': algorithm,'conference': conference}
    sig_b.append(query_dict)
    create_cache8()
    insert_cache8(sig_b)
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    paginated_results = sorted_b[start_index:end_index]
    return paginated_results,total_pages



@app.route('/paperb', methods=['GET'])
def sort_b():
    selected_paper = request.args.get('q')
    filter_option = request.args.get('filter')
    page = int(request.args.get('page', 1))
    results_per_page = int(request.args.get('results_per_page', 20))
    conference = request.args.get('conference')
    if not selected_paper:
        return jsonify([])
    else:
        if not conference:
            print("entered not cong")
            if filter_option == 'simple_algorithm':
                try:
                    paginated_results,total_pages = cache7(selected_paper,filter_option,page)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                query = retrieve_results_query(selected_paper)
                list_b = []
                for item in query:
                    t = []
                    y,c = yearsCitations(item[0])
                    if len(c) <= 0:
                        continue 
                    alpha = 1 / len(c)
                    cI,period = bestTimePeriod(y,c,alpha)
                    t.extend([item, y, c,period,cI])
                    list_b.append(t)
                sorted_b = sorted(list_b, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache7(page,selected_paper,filter_option,sorted_b,total_pages)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
            elif filter_option == 'exponential_dynamic':
                try:
                    paginated_results,total_pages = cache7(selected_paper,filter_option,page)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                sig_b = []
                query = retrieve_results_query(selected_paper)
                list_b = []
                for item in query:
                    t = []
                    y,c = yearsCitations(item[0])
                    if len(c) <= 0:
                        continue 
                    alpha = 1 / len(c)
                    cI,period = bestTimePeriodDynamic(y,c,alpha)
                    t.extend([item, y, c,period,cI])
                    list_b.append(t)
                sorted_b = sorted(list_b, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache7(page,selected_paper,filter_option,sorted_b,total_pages)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
                
                    
            elif filter_option == 'exponential_static':
                try:
                    paginated_results,total_pages = cache7(selected_paper,filter_option,page)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                query = retrieve_results_query(selected_paper)
                list_b = []
                for item in query:
                    t = []
                    y,c = yearsCitations(item[0])
                    if len(c) <= 0:
                        continue 
                    alpha = 0.9
                    cI,period = bestTimePeriod(y,c,alpha)
                    t.extend([item, y, c,period,cI])
                    list_b.append(t)
                sorted_b = sorted(list_b, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache7(page,selected_paper,filter_option,sorted_b,total_pages)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
        else:
            if filter_option == 'simple_algorithm':
                try:
                    paginated_results,total_pages = cache8(selected_paper,filter_option,page,conference)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                sig_b = []
                query = retrieve_results_query_conf(selected_paper,conference)
                list_b = []
                for item in query:
                    t = []
                    y,c = yearsCitations(item[0])
                    if len(c) <= 0:
                        continue 
                    alpha = 1 / len(c)
                    cI,period = bestTimePeriod(y,c,alpha)
                    t.extend([item, y, c,period,cI])
                    list_b.append(t)
                sorted_b = sorted(list_b, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache8(page,selected_paper,filter_option,sorted_b,total_pages,conference)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
            elif filter_option == 'exponential_dynamic':
                try:
                    paginated_results,total_pages = cache8(selected_paper,filter_option,page,conference)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                sig_b = []
                query = retrieve_results_query_conf(selected_paper,conference)
                list_b = []
                for item in query:
                    t = []
                    y,c = yearsCitations(item[0])
                    if len(c) <= 0:
                        continue 
                    alpha = 1 / len(c)
                    cI,period = bestTimePeriodDynamic(y,c,alpha)
                    t.extend([item, y, c,period,cI])
                    list_b.append(t)
                sorted_b = sorted(list_b, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache8(page,selected_paper,filter_option,sorted_b,total_pages,conference)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
                
                    
            elif filter_option == 'exponential_static':
                try:
                    paginated_results,total_pages = cache8(selected_paper,filter_option,page,conference)
                    return jsonify({
                        'results': paginated_results,
                        'totalPages': total_pages
                    })
                except Exception as e:
                    print("An exception occurred:", e)
                sig_b = []
                query = retrieve_results_query_conf(selected_paper,conference)
                list_b = []
                for item in query:
                    t = []
                    y,c = yearsCitations(item[0])
                    if len(c) <= 0:
                        continue 
                    alpha = 0.9
                    cI,period = bestTimePeriod(y,c,alpha)
                    t.extend([item, y, c,period,cI])
                    list_b.append(t)
                sorted_b = sorted(list_b, key=lambda x: x[-1], reverse=True)
                total_pages = int(len(query) / 20)
                paginated_results,total_pages = database_entry_cache8(page,selected_paper,filter_option,sorted_b,total_pages,conference)
                return jsonify({
                    'results': paginated_results,
                    'totalPages': total_pages
                })
                


@app.route('/generalStatistics.html')
def general_statistics():
    return render_template('generalStatistics.html',)


@app.route('/results.html')
def research_results():
    return render_template('results.html',)


@app.route('/paperdetails.html')
def paper_results():
    return render_template('paperdetails.html',)


@app.route('/test.html')
def n():
    conn = sqlite3.connect('complete_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT code from conferences")
    rows = cursor.fetchall()
    rows = [row[0] for row in rows]

    return render_template('test.html',confs = rows)

@app.route('/timeline.html')
def timeline():
    return render_template('timeline.html',)

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
