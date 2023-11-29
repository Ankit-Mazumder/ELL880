from sqlalchemy import create_engine,Table, Column, Integer, String, MetaData, insert,PrimaryKeyConstraint,NUMERIC,ForeignKey,UniqueConstraint

engine = create_engine('sqlite:///complete_data.db')
meta = MetaData()

#cache
def create_cache1():
    cache1 = Table("cache1", meta,
        Column('selected_paper', String, primary_key=True),
        Column('results', String),
        Column('total_pages', Integer),
        extend_existing=True
    )
    meta.create_all(engine)

def insert_cache1(row_values:dict)->None:
    cache1 = Table('cache1',meta)
    with engine.begin() as connection:
        connection.execute(cache1.insert(), row_values)

def create_cache2():
    cache2 = Table("cache2", meta,
        Column('selected_paper', String),
        Column('results', String),
        Column('total_pages', Integer),
        Column('conference', String),
        UniqueConstraint('selected_paper','conference'),
        extend_existing=True
    )
    meta.create_all(engine)

def insert_cache2(row_values:dict)->None:
    cache2 = Table('cache2',meta)
    with engine.begin() as connection:
        connection.execute(cache2.insert(), row_values)

def create_cache3():
    cache3 = Table("cache3", meta,
        Column('selected_paper', String, primary_key=True),
        Column('results', String),
        Column('total_pages', Integer),
        extend_existing=True
    )
    meta.create_all(engine)

def insert_cache3(row_values:dict)->None:
    cache3 = Table('cache3',meta)
    with engine.begin() as connection:
        connection.execute(cache3.insert(), row_values)


def create_cache4():
    cache4 = Table("cache4", meta,
        Column('selected_paper', String),
        Column('results', String),
        Column('total_pages', Integer),
        Column('conference', String),
        UniqueConstraint('selected_paper','conference'),
        extend_existing=True
    )
    meta.create_all(engine)

def insert_cache4(row_values:dict)->None:
    cache4 = Table('cache4',meta)
    with engine.begin() as connection:
        connection.execute(cache4.insert(), row_values)


def create_cache5():
    cache5 = Table("cache5", meta,
        Column('selected_paper', String),
        Column('results', String),
        Column('total_pages', Integer),
        Column('algorithm', String),
        UniqueConstraint('selected_paper','algorithm'),
        extend_existing=True
    )
    meta.create_all(engine)

def insert_cache5(row_values:dict)->None:
    cache5 = Table('cache5',meta)
    with engine.begin() as connection:
        connection.execute(cache5.insert(), row_values)



def create_cache6():
    cache6 = Table("cache6", meta,
        Column('selected_paper', String),
        Column('results', String),
        Column('total_pages', Integer),
        Column('conference', String),
        Column('algorithm', String),
        UniqueConstraint('selected_paper','conference','algorithm'),
        extend_existing=True
    )
    meta.create_all(engine)

def insert_cache6(row_values:dict)->None:
    cache6 = Table('cache6',meta)
    with engine.begin() as connection:
        connection.execute(cache6.insert(), row_values)

def create_cache7():
    cache7 = Table("cache7", meta,
        Column('selected_paper', String),
        Column('results', String),
        Column('total_pages', Integer),
        Column('algorithm', String),
        UniqueConstraint('selected_paper','algorithm'),
        extend_existing=True
    )
    meta.create_all(engine)

def insert_cache7(row_values:dict)->None:
    cache7 = Table('cache7',meta)
    with engine.begin() as connection:
        connection.execute(cache7.insert(), row_values)


def create_cache8():
    cache8 = Table("cache8", meta,
        Column('selected_paper', String),
        Column('results', String),
        Column('total_pages', Integer),
        Column('conference', String),
        Column('algorithm', String),
        UniqueConstraint('selected_paper','conference','algorithm'),
        extend_existing=True
    )
    meta.create_all(engine)

def insert_cache8(row_values:dict)->None:
    cache8 = Table('cache8',meta)
    with engine.begin() as connection:
        connection.execute(cache8.insert(), row_values)


def create_table():
    """ 
    This method creates a table in complete_data.db file with following schema:
    {paperId : string, title : string, conference : string, conference code : string, year : integer, citationCount : 
    integer }
    """


    rp_data = Table("rp_data",meta,
        Column('paperId',String),
        Column('title',String),
        Column('authors',String),
        Column('year',Integer),
        Column('citationCount',Integer),
        Column('publication', String)
    )
    meta.create_all(engine)

def create_main_table():
    """ 
    This method creates a table in complete_data.db file with following schema:
    {paperId : string, title : string, conference : string, conference code : string, year : integer, citationCount : 
    integer }
    """


    main_table = Table("main_table",meta,
        Column('paperId',String),
        Column('title',String),
        Column('authors',String),
        Column('year',Integer),
        Column('citationCount',Integer),
        Column('publication', String)
    )
    meta.create_all(engine)



def insert_main_table(row_values:dict)->None:
    main_table = Table('main_table',meta)
    with engine.begin() as connection:
        connection.execute(main_table.insert(), row_values)





def create_conf_table():

    conferences = Table("conferences",meta,
        Column('conference',String,primary_key=True),
        Column('code',String),
        Column('ranking',String),
        Column('dblp_link',String)
    )
    meta.create_all(engine)

def create_events_table():
    Events = Table("Events",meta,
        Column('code',String),
        Column('year',Integer),
        Column('link',String)
    )
    meta.create_all(engine)




def create_papers_links_table():
    rp_links = Table("rp_links",meta,
        Column('code',String),
        Column('year',Integer),
        Column('title',String),
        Column('link',String)
    )
    meta.create_all(engine)

def create_table2():
    table2 = Table("table2",meta,
        Column('paper_id',String),
        Column('year',Integer),
        Column('citation_count',Integer),
        PrimaryKeyConstraint('paper_id','year','citation_count')
    )
    meta.create_all(engine)


def create_cited_papers():
    cited_papers = Table("cited_papers",meta,
        Column('paper_id',String),
        Column('cited_paper',String),
        Column('venue', Integer),
        PrimaryKeyConstraint('paper_id','cited_paper','venue')
    )
    meta.create_all(engine)

def cited_papers_n():
    cited_papers = Table("cited_papers_n",meta,
        Column('paper_id',String),
        Column('cited_paper',String),
        Column('venue', Integer),
        PrimaryKeyConstraint('paper_id','cited_paper','venue')
    )
    meta.create_all(engine)

def superSetAnalysis():
    superSetAnalysis = Table("superSetAnalysis",meta,
        Column('abstract',String),
        Column('authors',String),
        Column('citationVelocity',Integer),
        Column('citations', String),
        Column('corpusId', Integer),
        Column('doi', String),
        Column('fieldsOfStudy', String),
        Column('influentialCitationCount', Integer),
        Column('isOpenAccess',String),
        Column('isPublisherLicence', String),
        Column('numCitebBy',Integer),
        Column('numCiting', Integer),
        Column('paperId', String),
        Column('references', String),
        Column('title',String),
        Column('topics', String),
        Column('venue', String),
        Column('year', Integer),
        PrimaryKeyConstraint('paperId')
    )
    meta.create_all(engine)

def insert_superSetAnalysis(row_values:dict)->None:
    superSetAnalysis = Table('superSetAnalysis',meta)
    with engine.begin() as connection:
        connection.execute(superSetAnalysis.insert(), row_values)



def insert_cited_papers_n(row_values:dict)->None:
    cited_papers = Table('cited_papers_n',meta)
    with engine.begin() as connection:
        connection.execute(cited_papers.insert(), row_values)




def insert_events(row_values:dict)->None:
    Events = Table('Events',meta)
    with engine.begin() as connection:
        connection.execute(Events.insert(), row_values)

def insert_conferences(row_values:dict)->None:
    conferences = Table('conferences',meta)
    with engine.begin() as connection:
        connection.execute(conferences.insert(), row_values)

def insert_rows(row_values:dict)->None:
    rp_data = Table('rp_data',meta)
    with engine.begin() as connection:
        connection.execute(rp_data.insert(), row_values)

def insert_rp_links(row_values:dict)->None:
    rp_links = Table('rp_links',meta)
    with engine.begin() as connection:
        connection.execute(rp_links.insert(), row_values)

def insert_table2(row_values:dict)->None:
    table2 = Table('table2',meta)
    with engine.begin() as connection:
        connection.execute(table2.insert(), row_values)

def insert_cited_papers(row_values:dict)->None:
    cited_papers = Table('cited_papers',meta)
    with engine.begin() as connection:
        connection.execute(cited_papers.insert(), row_values)

